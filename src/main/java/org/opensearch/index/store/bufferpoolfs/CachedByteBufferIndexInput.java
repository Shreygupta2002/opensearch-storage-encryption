/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.index.store.bufferpoolfs;

import static org.opensearch.index.store.bufferpoolfs.StaticConfigs.CACHE_BLOCK_MASK;
import static org.opensearch.index.store.bufferpoolfs.StaticConfigs.CACHE_BLOCK_SIZE;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.opensearch.index.store.block.RefCountedByteBuffer;
import org.opensearch.index.store.block_cache.BlockCache;
import org.opensearch.index.store.block_cache.BlockCacheValue;
import org.opensearch.index.store.block_cache.FileBlockCacheKey;
import org.opensearch.index.store.read_ahead.ReadaheadContext;
import org.opensearch.index.store.read_ahead.ReadaheadManager;

/**
 * IndexInput backed by direct {@link ByteBuffer}s with block-level caching.
 * GC-managed lifecycle — no reference counting, no pin/unpin overhead.
 *
 * <p>Uses {@link BlockSlotTinyCache<RefCountedByteBuffer>} for L1 and {@link BlockCache} (Caffeine) for L2.
 * All reads use absolute ByteBuffer methods (thread-safe, no position mutation).
 *
 * @opensearch.internal
 */
public class CachedByteBufferIndexInput extends IndexInput implements RandomAccessInput {

    final long length;
    final Path path;
    final BlockCache<RefCountedByteBuffer> blockCache;
    final ReadaheadManager readaheadManager;
    final ReadaheadContext readaheadContext;
    final long absoluteBaseOffset;
    final boolean isSlice;
    /** Opened with {@code IOContext.READONCE}: its reads leave the cache unchanged, like a merge's. */
    final boolean readOnce;

    long curPosition = 0L;
    volatile boolean isOpen = true;

    /** Lookups to retry when the block is evicted between the cache lookup and reading its buffer. */
    private static final int MAX_CLOSED_BLOCK_RETRIES = 10;

    private long currentBlockOffset = -1;
    private BlockCacheValue<RefCountedByteBuffer> currentBlock = null;
    private int lastOffsetInBlock;
    private int currentBlockLength; // actual data length in current block (may be < capacity for last block)

    private final BlockSlotTinyCache<RefCountedByteBuffer> slotCache;
    private final BlockSlotTinyCache.CacheHitHolder cacheHitHolder = new BlockSlotTinyCache.CacheHitHolder();

    /*
     * Reads outside the cache, by a merge or read-once input (see MergeCachePolicy), go into this input's own
     * decrypted chunks. A read that continues where a chunk ends doubles that chunk's next read, from
     * UNCACHED_MIN_READ_BYTES up to UNCACHED_MAX_READ_BYTES; any other read starts small again, so a clone or
     * a reader that jumps around reads little more than the blocks it uses. Two chunks, so that a reader
     * alternating between two parts of the file doesn't reread them.
     */
    static final int UNCACHED_MIN_READ_BYTES = 16 << 10;
    static final int UNCACHED_MAX_READ_BYTES = 256 << 10;
    private final byte[][] chunkData = new byte[2][];
    private final long[] chunkStart = { -1, -1 };
    private final int[] chunkLength = new int[2];
    private final int[] chunkNextRead = new int[2];
    private int lastChunk;
    /** The current block when it was read outside the cache (then {@link #currentBlock} is null). */
    private ByteBuffer currentUncachedBlock;

    public static CachedByteBufferIndexInput newInstance(
        String resourceDescription,
        Path path,
        long length,
        BlockCache<RefCountedByteBuffer> blockCache,
        ReadaheadManager readaheadManager,
        ReadaheadContext readaheadContext,
        BlockSlotTinyCache<RefCountedByteBuffer> slotCache
    ) {
        return newInstance(resourceDescription, path, length, blockCache, readaheadManager, readaheadContext, slotCache, false);
    }

    /** As above; a {@code readOnce} input's reads leave the cache unchanged (see {@link MergeCachePolicy}). */
    public static CachedByteBufferIndexInput newInstance(
        String resourceDescription,
        Path path,
        long length,
        BlockCache<RefCountedByteBuffer> blockCache,
        ReadaheadManager readaheadManager,
        ReadaheadContext readaheadContext,
        BlockSlotTinyCache<RefCountedByteBuffer> slotCache,
        boolean readOnce
    ) {
        CachedByteBufferIndexInput input = new CachedByteBufferIndexInput(
            resourceDescription,
            path,
            0,
            length,
            blockCache,
            readaheadManager,
            readaheadContext,
            false,
            slotCache,
            readOnce
        );
        try {
            input.seek(0L);
        } catch (IOException ioe) {
            throw new AssertionError(ioe);
        }
        return input;
    }

    private CachedByteBufferIndexInput(
        String resourceDescription,
        Path path,
        long absoluteBaseOffset,
        long length,
        BlockCache<RefCountedByteBuffer> blockCache,
        ReadaheadManager readaheadManager,
        ReadaheadContext readaheadContext,
        boolean isSlice,
        BlockSlotTinyCache<RefCountedByteBuffer> slotCache,
        boolean readOnce
    ) {
        super(resourceDescription);
        this.readOnce = readOnce;
        this.path = path;
        this.absoluteBaseOffset = absoluteBaseOffset;
        this.length = length;
        this.blockCache = blockCache;
        this.readaheadManager = readaheadManager;
        this.readaheadContext = readaheadContext;
        this.isSlice = isSlice;
        this.slotCache = slotCache;
    }

    void ensureOpen() {
        if (!isOpen)
            throw alreadyClosed(null);
    }

    RuntimeException handlePositionalIOOBE(RuntimeException unused, String action, long pos) throws IOException {
        if (pos < 0L)
            return new IllegalArgumentException(action + " negative position (pos=" + pos + "): " + this);
        else
            throw new EOFException(action + " past EOF (pos=" + pos + "): " + this);
    }

    AlreadyClosedException alreadyClosed(RuntimeException unused) {
        return new AlreadyClosedException("Already closed: " + this);
    }

    private ByteBuffer getCacheBlockWithOffset(long pos) throws IOException {
        final long fileOffset = absoluteBaseOffset + pos;
        final long blockOffset = fileOffset & ~CACHE_BLOCK_MASK;
        final int offsetInBlock = (int) (fileOffset - blockOffset);

        if (blockOffset == currentBlockOffset) {
            if (currentUncachedBlock != null) {
                lastOffsetInBlock = offsetInBlock;
                return currentUncachedBlock;
            }
            if (currentBlock != null) {
                final ByteBuffer buffer = currentBlock.value().buffer();
                if (buffer != null) {
                    lastOffsetInBlock = offsetInBlock;
                    return buffer;
                }
                // The cache has dropped this block since the last read (its buffer is cleared): look it up again.
            }
        }

        if (MergeCachePolicy.bypassReads(readOnce) && blockCache.supportsUncachedReads()) {
            return getBlockLeavingCacheUnchanged(blockOffset, offsetInBlock);
        }
        currentUncachedBlock = null;

        BlockCacheValue<RefCountedByteBuffer> cacheValue;
        ByteBuffer buffer;
        int attempts = 0;
        do {
            if (++attempts > MAX_CLOSED_BLOCK_RETRIES) {
                throw new IOException("Block at offset " + blockOffset + " was evicted on each of " + MAX_CLOSED_BLOCK_RETRIES + " reads");
            }
            cacheHitHolder.reset();
            cacheValue = slotCache.acquireRefCountedValue(blockOffset, cacheHitHolder);
            if (cacheValue == null) {
                throw new IOException("Failed to acquire cache value for block at offset " + blockOffset);
            }
            // Null if the cache dropped the block between the lookup and here.
            buffer = cacheValue.value().buffer();
        } while (buffer == null);

        currentBlockOffset = blockOffset;
        currentBlock = cacheValue;
        currentBlockLength = cacheValue.length();

        if (readaheadContext != null) {
            readaheadContext.onAccess(blockOffset, cacheHitHolder.wasCacheHit());
        }

        lastOffsetInBlock = offsetInBlock;
        return buffer;
    }

    /**
     * Returns the block at {@code blockOffset} without changing the cache: a cached copy if there is one,
     * otherwise this input's own decrypted copy. Used for merges and read-once inputs, which read each block
     * once (see {@link MergeCachePolicy}).
     */
    private ByteBuffer getBlockLeavingCacheUnchanged(long blockOffset, int offsetInBlock) throws IOException {
        int slot = chunkHolding(blockOffset);
        if (slot < 0) {
            final BlockCacheValue<RefCountedByteBuffer> cached = blockCache.peek(new FileBlockCacheKey(path, blockOffset));
            final ByteBuffer cachedBuffer = cached != null ? cached.value().buffer() : null;
            if (cachedBuffer != null) {
                MergeCachePolicy.onReadFromCache();
                currentUncachedBlock = null;
                currentBlockOffset = blockOffset;
                currentBlock = cached;
                currentBlockLength = cached.length();
                lastOffsetInBlock = offsetInBlock;
                return cachedBuffer;
            }
            slot = fillChunk(blockOffset);
        }
        lastChunk = slot;
        final int indexInChunk = (int) (blockOffset - chunkStart[slot]);
        final int blockLength = Math.min(CACHE_BLOCK_SIZE, chunkLength[slot] - indexInChunk);
        currentUncachedBlock = ByteBuffer.wrap(chunkData[slot], indexInChunk, blockLength).slice().order(ByteOrder.LITTLE_ENDIAN);
        currentBlock = null;
        currentBlockOffset = blockOffset;
        currentBlockLength = blockLength;
        lastOffsetInBlock = offsetInBlock;
        return currentUncachedBlock;
    }

    /** The chunk holding {@code blockOffset}, or -1. */
    private int chunkHolding(long blockOffset) {
        for (int i = 0; i < 2; i++) {
            final int s = i == 0 ? lastChunk : 1 - lastChunk;
            if (chunkStart[s] >= 0 && blockOffset >= chunkStart[s] && blockOffset < chunkStart[s] + chunkLength[s]) {
                return s;
            }
        }
        return -1;
    }

    /** Reads the chunk starting at {@code blockOffset} and returns its slot. */
    private int fillChunk(long blockOffset) throws IOException {
        int slot = -1;
        int size = UNCACHED_MIN_READ_BYTES;
        for (int i = 0; i < 2; i++) {
            final int s = i == 0 ? lastChunk : 1 - lastChunk;
            if (chunkStart[s] >= 0 && blockOffset == chunkStart[s] + chunkLength[s]) {
                slot = s; // continues this chunk: read more at once
                size = chunkNextRead[s];
                break;
            }
        }
        if (slot < 0) {
            slot = 1 - lastChunk; // a jump: replace the chunk used less recently, and start small
        }
        // Read up to the end of this input's last block at most; the file may be longer (a slice) or end sooner.
        final long end = (absoluteBaseOffset + length + CACHE_BLOCK_MASK) & ~CACHE_BLOCK_MASK;
        final int toRead = (int) Math.min(size, end - blockOffset);
        if (chunkData[slot] == null || chunkData[slot].length < toRead) {
            chunkData[slot] = new byte[toRead];
        }
        chunkStart[slot] = -1;
        final int read = blockCache.readUncached(path, blockOffset, chunkData[slot], 0, toRead);
        if (read <= 0) {
            throw new EOFException("No data at offset " + blockOffset + ": " + this);
        }
        MergeCachePolicy.onUncachedRead(read);
        chunkStart[slot] = blockOffset;
        chunkLength[slot] = read;
        chunkNextRead[slot] = Math.min(UNCACHED_MAX_READ_BYTES, Math.max(UNCACHED_MIN_READ_BYTES, size * 2));
        return slot;
    }

    @Override
    public final byte readByte() throws IOException {
        final long currentPos = curPosition;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(currentPos);
            final byte v = buf.get(lastOffsetInBlock);
            curPosition = currentPos + 1;
            return v;
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", currentPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public final void readBytes(byte[] b, int offset, int len) throws IOException {
        if (len == 0)
            return;
        final long startPos = curPosition;
        int remaining = len;
        int bufferOffset = offset;
        long currentPos = startPos;

        try {
            while (remaining > 0) {
                final ByteBuffer buf = getCacheBlockWithOffset(currentPos);
                final int offInBlock = lastOffsetInBlock;
                final int avail = currentBlockLength - offInBlock;

                if (offInBlock == 0 && remaining >= CACHE_BLOCK_SIZE && currentBlockLength >= CACHE_BLOCK_SIZE) {
                    buf.get(0, b, bufferOffset, CACHE_BLOCK_SIZE);
                    remaining -= CACHE_BLOCK_SIZE;
                    bufferOffset += CACHE_BLOCK_SIZE;
                    currentPos += CACHE_BLOCK_SIZE;
                    continue;
                }

                final int toRead = Math.min(remaining, avail);
                buf.get(offInBlock, b, bufferOffset, toRead);
                remaining -= toRead;
                bufferOffset += toRead;
                currentPos += toRead;
            }
            curPosition = startPos + len;
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", startPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public void readInts(int[] dst, int offset, int length) throws IOException {
        if (length == 0)
            return;
        final long startPos = getFilePointer();
        final long totalBytes = Integer.BYTES * (long) length;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(startPos);
            final int offInBlock = lastOffsetInBlock;
            if (offInBlock + totalBytes <= currentBlockLength) {
                for (int i = 0; i < length; i++) {
                    dst[offset + i] = buf.getInt(offInBlock + i * Integer.BYTES);
                }
                curPosition += totalBytes;
            } else {
                super.readInts(dst, offset, length);
            }
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", startPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public void readLongs(long[] dst, int offset, int length) throws IOException {
        if (length == 0)
            return;
        final long startPos = getFilePointer();
        final long totalBytes = Long.BYTES * (long) length;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(startPos);
            final int offInBlock = lastOffsetInBlock;
            if (offInBlock + totalBytes <= currentBlockLength) {
                for (int i = 0; i < length; i++) {
                    dst[offset + i] = buf.getLong(offInBlock + i * Long.BYTES);
                }
                curPosition += totalBytes;
            } else {
                super.readLongs(dst, offset, length);
            }
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", startPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public void readFloats(float[] dst, int offset, int length) throws IOException {
        if (length == 0)
            return;
        final long startPos = getFilePointer();
        final long totalBytes = Float.BYTES * (long) length;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(startPos);
            final int offInBlock = lastOffsetInBlock;
            if (offInBlock + totalBytes <= currentBlockLength) {
                for (int i = 0; i < length; i++) {
                    dst[offset + i] = buf.getFloat(offInBlock + i * Float.BYTES);
                }
                curPosition += totalBytes;
            } else {
                super.readFloats(dst, offset, length);
            }
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", startPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public final short readShort() throws IOException {
        final long currentPos = getFilePointer();
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(currentPos);
            if (lastOffsetInBlock + Short.BYTES > currentBlockLength)
                return super.readShort();
            final short v = buf.getShort(lastOffsetInBlock);
            curPosition += Short.BYTES;
            return v;
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", currentPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public final int readInt() throws IOException {
        final long currentPos = curPosition;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(currentPos);
            if (lastOffsetInBlock + Integer.BYTES > currentBlockLength)
                return super.readInt();
            final int v = buf.getInt(lastOffsetInBlock);
            curPosition = currentPos + Integer.BYTES;
            return v;
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", currentPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public final long readLong() throws IOException {
        final long currentPos = curPosition;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(currentPos);
            if (lastOffsetInBlock + Long.BYTES > currentBlockLength)
                return super.readLong();
            final long v = buf.getLong(lastOffsetInBlock);
            curPosition = currentPos + Long.BYTES;
            return v;
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", currentPos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public final int readVInt() throws IOException {
        return super.readVInt();
    }

    @Override
    public final long readVLong() throws IOException {
        return super.readVLong();
    }

    @Override
    public long getFilePointer() {
        ensureOpen();
        return curPosition;
    }

    @Override
    public void seek(long pos) throws IOException {
        ensureOpen();
        if (pos < 0 || pos > length)
            throw handlePositionalIOOBE(null, "seek", pos);
        this.curPosition = pos;
    }

    @Override
    public byte readByte(long pos) throws IOException {
        if (pos < 0 || pos >= length)
            return 0;
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(pos);
            return buf.get(lastOffsetInBlock);
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", pos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public short readShort(long pos) throws IOException {
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(pos);
            if (lastOffsetInBlock + Short.BYTES > currentBlockLength) {
                long savedPos = getFilePointer();
                try {
                    seek(pos);
                    return readShort();
                } finally {
                    seek(savedPos);
                }
            }
            return buf.getShort(lastOffsetInBlock);
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", pos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public int readInt(long pos) throws IOException {
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(pos);
            if (lastOffsetInBlock + Integer.BYTES > currentBlockLength) {
                long savedPos = getFilePointer();
                try {
                    seek(pos);
                    return readInt();
                } finally {
                    seek(savedPos);
                }
            }
            return buf.getInt(lastOffsetInBlock);
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", pos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public long readLong(long pos) throws IOException {
        try {
            final ByteBuffer buf = getCacheBlockWithOffset(pos);
            if (lastOffsetInBlock + Long.BYTES > currentBlockLength) {
                long savedPos = getFilePointer();
                try {
                    seek(pos);
                    return readLong();
                } finally {
                    seek(savedPos);
                }
            }
            return buf.getLong(lastOffsetInBlock);
        } catch (IndexOutOfBoundsException ioobe) {
            throw handlePositionalIOOBE(ioobe, "read", pos);
        } catch (NullPointerException | IllegalStateException e) {
            throw alreadyClosed(e);
        }
    }

    @Override
    public final long length() {
        return length;
    }

    @Override
    public final CachedByteBufferIndexInput clone() {
        final CachedByteBufferIndexInput clone = buildSlice(null, 0L, this.length);
        try {
            clone.seek(getFilePointer());
        } catch (IOException ioe) {
            throw new AssertionError(ioe);
        }
        return clone;
    }

    @Override
    public final CachedByteBufferIndexInput slice(String sliceDescription, long offset, long length) throws IOException {
        if (offset < 0 || length < 0 || offset + length > this.length) {
            throw new IllegalArgumentException(
                "slice() "
                    + sliceDescription
                    + " out of bounds: offset="
                    + offset
                    + ",length="
                    + length
                    + ",fileLength="
                    + this.length
                    + ": "
                    + this
            );
        }
        var slice = buildSlice(sliceDescription, offset, length);
        slice.seek(0L);
        return slice;
    }

    CachedByteBufferIndexInput buildSlice(String sliceDescription, long sliceOffset, long length) {
        ensureOpen();
        final long sliceAbsoluteBaseOffset = this.absoluteBaseOffset + sliceOffset;
        final String newResourceDescription = getFullSliceDescription(sliceDescription);
        CachedByteBufferIndexInput slice = new CachedByteBufferIndexInput(
            newResourceDescription,
            path,
            sliceAbsoluteBaseOffset,
            length,
            blockCache,
            readaheadManager,
            readaheadContext,
            true,
            slotCache,
            readOnce
        );
        try {
            slice.seek(0L);
        } catch (IOException ioe) {
            throw new AssertionError(ioe);
        }
        return slice;
    }

    @Override
    public final void close() throws IOException {
        if (!isOpen)
            return;
        isOpen = false;
        currentBlock = null;
        currentUncachedBlock = null;
        chunkData[0] = chunkData[1] = null;
        currentBlockOffset = -1;

        if (!isSlice) {
            if (slotCache != null)
                slotCache.clear();
            readaheadManager.close();
        }
    }
}
