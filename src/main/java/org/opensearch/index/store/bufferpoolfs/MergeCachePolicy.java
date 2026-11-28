/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.index.store.bufferpoolfs;

import static org.opensearch.index.store.bufferpoolfs.StaticConfigs.CACHE_BLOCK_SIZE;

import java.util.concurrent.atomic.AtomicLong;

import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.store.IOContext;
import org.opensearch.index.store.CryptoDirectoryFactory;

/**
 * Keeps one-pass I/O out of the block cache, so a large merge (such as the force merge at the end of an
 * index build) doesn't evict the blocks searches are using.
 *
 * <ul>
 * <li><b>Reads by a merge, or to copy files to another shard copy.</b> A merge reads each block of its source
 * segments once, and those segments are deleted when it finishes; peer recovery (for example, adding replicas to
 * a freshly built index) and segment replication read each file once to send it. Neither carries a telling
 * {@link IOContext}: Lucene merges through the readers it opened for search, and recovery opens files with the
 * default context. So they are recognised by thread: Lucene's merge threads and OpenSearch's generic pool. Such
 * a read uses a cached block if there is one, and otherwise reads and decrypts into the reader's own buffer
 * without adding the block to the cache. {@link IOContext#READONCE} inputs are treated the same way.</li>
 * <li><b>Writes.</b> The write cache keeps a plaintext copy of each written block so that new segments are
 * warm when searches reach them. For a merge whose estimated size is at least {@link #largeWriteBytes} of the
 * cache, nothing is cached: its output would only push out the blocks searches use. Nor is a file copied
 * in from another shard copy by peer recovery or segment replication (the receiving node writes those
 * chunks on the generic pool): it is written under a temporary name ({@code recovery.<id>.<name>},
 * {@code replication.<id>.<name>}) and renamed when the copy completes, and the cache is keyed by path, so
 * blocks cached while writing would never be read. Every other file is cached up to
 * {@link #largeWriteBytes}.</li>
 * </ul>
 *
 * Enabled by {@code node.store.crypto.merge_cache_bypass_enabled} (default true, dynamic).
 *
 * @opensearch.internal
 */
public final class MergeCachePolicy {

    /** Fraction of the cache's capacity at or above which a merge, or the rest of a file, isn't write-cached. */
    static final double LARGE_WRITE_FRACTION = 0.02;
    /** Lower bound on that size, so that a small cache still write-caches ordinary segments. */
    static final long MIN_LARGE_WRITE_BYTES = 16L << 20;

    private static final AtomicLong readBlocksFromCache = new AtomicLong();
    private static final AtomicLong readBytesUncached = new AtomicLong();
    private static final AtomicLong uncachedReads = new AtomicLong();
    private static final AtomicLong writeBlocksNotCached = new AtomicLong();
    private static final AtomicLong largeMergeOutputs = new AtomicLong();
    private static final AtomicLong copyOutputs = new AtomicLong();

    private MergeCachePolicy() {}

    public static boolean enabled() {
        return CryptoDirectoryFactory.isMergeCacheBypassEnabled();
    }

    /** Whether reads on the current thread should leave the cache unchanged. */
    static boolean bypassReads(boolean readOnceInput) {
        return enabled() && (readOnceInput || BULK_READ_THREAD.get());
    }

    /** Per thread: whether it reads files once in bulk ({@link #isBulkReadThread}); a thread's role doesn't change. */
    private static final ThreadLocal<Boolean> BULK_READ_THREAD = ThreadLocal.withInitial(() -> isBulkReadThread(Thread.currentThread()));

    /**
     * Whether {@code thread} reads files once, in bulk: a Lucene merge thread, or an OpenSearch {@code generic}
     * pool thread, which is where peer recovery and segment replication read the files they send to other
     * copies (their file-chunk responses are handled on the generic pool, and the files are opened with
     * {@link IOContext#DEFAULT}, so the context doesn't identify them). Searches run on the search pools.
     */
    static boolean isBulkReadThread(Thread thread) {
        return isMergeThread(thread) || isGenericThread(thread);
    }

    /**
     * Whether {@code thread} is in OpenSearch's {@code generic} pool. On the sending node it reads files for
     * peer recovery and segment replication; on the receiving node it writes the file chunks (the
     * {@code recovery/file_chunk}, {@code replication/file_chunk} and {@code merged_segment_file_chunk}
     * handlers all run there).
     */
    static boolean isGenericThread(Thread thread) {
        return thread.getName().contains("[generic]");
    }

    /** Whether {@code thread} runs Lucene merges: a {@code ConcurrentMergeScheduler.MergeThread}. */
    static boolean isMergeThread(Thread thread) {
        return MERGE_THREAD_CLASS != null && MERGE_THREAD_CLASS.isInstance(thread);
    }

    /** {@code ConcurrentMergeScheduler.MergeThread}, which is protected, so it is looked up by name. */
    private static final Class<?> MERGE_THREAD_CLASS = findMergeThreadClass();

    private static Class<?> findMergeThreadClass() {
        try {
            return Class
                .forName(ConcurrentMergeScheduler.class.getName() + "$MergeThread", false, ConcurrentMergeScheduler.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            return null;
        }
    }

    /**
     * Size at or above which writes aren't cached: {@link #LARGE_WRITE_FRACTION} of a cache of
     * {@code cacheMaxBlocks} blocks, at least {@link #MIN_LARGE_WRITE_BYTES}. Unbounded for an unbounded cache.
     */
    static long largeWriteBytes(long cacheMaxBlocks) {
        if (cacheMaxBlocks <= 0) {
            return Long.MAX_VALUE;
        }
        return Math.max(MIN_LARGE_WRITE_BYTES, (long) (cacheMaxBlocks * (double) CACHE_BLOCK_SIZE * LARGE_WRITE_FRACTION));
    }

    /**
     * How many leading bytes of a file written with {@code context} on the current thread may go into the
     * write cache: none for a merge estimated at {@code largeWriteBytes} or more or for a file copied in on
     * the generic pool, otherwise {@code largeWriteBytes}.
     */
    static long writeCacheLimit(IOContext context, long largeWriteBytes) {
        if (enabled() == false) {
            return Long.MAX_VALUE;
        }
        if (context != null
            && context.context() == IOContext.Context.MERGE
            && context.mergeInfo() != null
            && context.mergeInfo().estimatedMergeBytes() >= largeWriteBytes) {
            largeMergeOutputs.incrementAndGet();
            return 0;
        }
        if (isGenericThread(Thread.currentThread())) {
            copyOutputs.incrementAndGet();
            return 0;
        }
        return largeWriteBytes;
    }

    static void onReadFromCache() {
        readBlocksFromCache.incrementAndGet();
    }

    static void onUncachedRead(long bytes) {
        readBytesUncached.addAndGet(bytes);
        uncachedReads.incrementAndGet();
    }

    static void onWriteBlockNotCached() {
        writeBlocksNotCached.incrementAndGet();
    }

    public static long readBlocksFromCache() {
        return readBlocksFromCache.get();
    }

    public static long readBytesUncached() {
        return readBytesUncached.get();
    }

    /** Reads from disk made outside the cache (each up to {@code CachedByteBufferIndexInput.UNCACHED_MAX_READ_BYTES}). */
    public static long uncachedReads() {
        return uncachedReads.get();
    }

    public static long writeBlocksNotCached() {
        return writeBlocksNotCached.get();
    }

    public static long largeMergeOutputs() {
        return largeMergeOutputs.get();
    }

    /** Files written on the generic pool (copied in from another shard copy), not write-cached. */
    public static long copyOutputs() {
        return copyOutputs.get();
    }

    /** One-line summary for the periodic memory log. */
    public static String stats() {
        return String
            .format(
                java.util.Locale.ROOT,
                "MergeCacheBypass[enabled=%b, readBlocksFromCache=%d, readUncachedMB=%d, uncachedReads=%d, writeBlocksNotCached=%d, "
                    + "largeMergeOutputs=%d, copyOutputs=%d]",
                enabled(),
                readBlocksFromCache.get(),
                readBytesUncached.get() >> 20,
                uncachedReads.get(),
                writeBlocksNotCached.get(),
                largeMergeOutputs.get(),
                copyOutputs.get()
            );
    }
}
