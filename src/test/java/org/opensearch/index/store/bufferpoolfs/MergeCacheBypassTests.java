/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.index.store.bufferpoolfs;

import static org.mockito.Mockito.mock;
import static org.opensearch.index.store.CryptoDirectoryFactory.DEFAULT_CRYPTO_PROVIDER;
import static org.opensearch.index.store.bufferpoolfs.StaticConfigs.CACHE_BLOCK_SIZE;

import java.nio.file.Path;
import java.security.Provider;
import java.security.Security;
import java.util.HashSet;
import java.util.Random;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FSLockFactory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MergeInfo;
import org.junit.After;
import org.junit.Before;
import org.opensearch.common.crypto.MasterKeyProvider;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexModule;
import org.opensearch.index.store.CryptoDirectoryFactory;
import org.opensearch.index.store.DummyKeyProvider;
import org.opensearch.index.store.block.RefCountedByteBuffer;
import org.opensearch.index.store.block_cache.BlockCache;
import org.opensearch.index.store.block_cache.CaffeineBlockCache;
import org.opensearch.index.store.block_cache.FileBlockCacheKey;
import org.opensearch.index.store.block_loader.BlockLoader;
import org.opensearch.index.store.block_loader.CryptoDirectIOBlockLoader;
import org.opensearch.index.store.cipher.EncryptionMetadataCache;
import org.opensearch.index.store.cipher.EncryptionMetadataCacheRegistry;
import org.opensearch.index.store.hybrid.HybridCryptoDirectory;
import org.opensearch.index.store.key.KeyResolver;
import org.opensearch.index.store.metrics.CryptoMetricsService;
import org.opensearch.index.store.pool.Pool;
import org.opensearch.index.store.pool.PoolBuilder;
import org.opensearch.telemetry.metrics.MetricsRegistry;
import org.opensearch.test.OpenSearchTestCase;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;

/**
 * {@link MergeCachePolicy} in ByteBuffer mode with the slab pool: merges, read-once inputs and large
 * writes leave the block cache as it was, and read back the right data.
 */
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
@SuppressWarnings("unchecked")
public class MergeCacheBypassTests extends OpenSearchTestCase {

    private Path path;
    private BufferPoolDirectory directory;
    /** What a shard uses: segments_N and other small files go to NIO, the rest to {@link #directory}. */
    private Directory hybrid;
    private PoolBuilder.PoolResources poolResources;
    private CaffeineBlockCache<RefCountedByteBuffer, RefCountedByteBuffer> sharedCache;
    private long largeWriteBytes;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        CryptoMetricsService.initialize(mock(MetricsRegistry.class));
        setBypass(true);
        path = createTempDir();
        Provider provider = Security.getProvider(DEFAULT_CRYPTO_PROVIDER);
        MasterKeyProvider keyProvider = DummyKeyProvider.create();
        // The minimum pool (256 MB), with a cache of ~200 MB: the large-write size is its 16 MB floor.
        poolResources = PoolBuilder
            .build(
                Settings
                    .builder()
                    .put("plugins.crypto.enabled", true)
                    .put("node.store.crypto.pool_size_percentage", 0.0)
                    .put("node.store.crypto.warmup_percentage", 0.0)
                    .put("node.store.crypto.cache_to_pool_ratio", 0.8)
                    .build()
            );
        Pool<RefCountedByteBuffer> pool = (Pool<RefCountedByteBuffer>) poolResources.getPool();

        String indexUuid = randomAlphaOfLength(10);
        String indexName = randomAlphaOfLength(10);
        KeyResolver keyResolver = new TestKeyResolver(indexUuid, indexName, FSDirectory.open(path), provider, keyProvider, 0);
        EncryptionMetadataCache metadataCache = EncryptionMetadataCacheRegistry.getOrCreateCache(indexUuid, 0, indexName);
        BlockLoader<RefCountedByteBuffer> loader = new CryptoDirectIOBlockLoader<>(
            pool,
            keyResolver,
            metadataCache,
            poolResources.getFileChannelCache(),
            true
        );
        sharedCache = (CaffeineBlockCache<RefCountedByteBuffer, RefCountedByteBuffer>) poolResources.getBlockCache();
        BlockCache<RefCountedByteBuffer> cache = new CaffeineBlockCache<>(
            sharedCache.getCache(),
            loader,
            poolResources.getMaxCacheBlocks()
        );
        largeWriteBytes = MergeCachePolicy.largeWriteBytes(cache.maxBlocks());
        assertEquals(MergeCachePolicy.MIN_LARGE_WRITE_BYTES, largeWriteBytes);
        directory = new BufferPoolDirectory(
            path,
            FSLockFactory.getDefault(),
            provider,
            keyResolver,
            pool,
            cache,
            poolResources.getSharedReadaheadWorker(),
            metadataCache,
            poolResources.getFileChannelCache(),
            true
        );
        hybrid = new HybridCryptoDirectory(
            FSLockFactory.getDefault(),
            directory,
            provider,
            keyResolver,
            metadataCache,
            new HashSet<>(IndexModule.INDEX_STORE_HYBRID_NIO_EXTENSIONS.getDefault(Settings.EMPTY))
        );
    }

    @After
    public void tearDown() throws Exception {
        CryptoDirectoryFactory.setNodeSettings(Settings.EMPTY);
        hybrid.close(); // also closes directory
        poolResources.close();
        super.tearDown();
    }

    private static void setBypass(boolean enabled) {
        CryptoDirectoryFactory
            .setNodeSettings(
                Settings
                    .builder()
                    .put("node.store.crypto.write_cache_enabled", true)
                    .put("node.store.crypto.byte_buffer_mode_enabled", true)
                    .put("node.store.crypto.byte_buffer_slab_pool_enabled", true)
                    .put("node.store.crypto.merge_cache_bypass_enabled", enabled)
                    .build()
            );
    }

    private static byte expected(long offset) {
        return (byte) ((offset * 31) ^ (offset >>> 13));
    }

    private void writeFile(String name, IOContext context, long bytes) throws Exception {
        byte[] chunk = new byte[1 << 20];
        try (IndexOutput out = directory.createOutput(name, context)) {
            for (long base = 0; base < bytes; base += chunk.length) {
                int n = (int) Math.min(chunk.length, bytes - base);
                for (int i = 0; i < n; i++) {
                    chunk[i] = expected(base + i);
                }
                out.writeBytes(chunk, n);
            }
        }
    }

    private void assertFileReadsBack(IndexInput in, long bytes) throws Exception {
        assertEquals(bytes, in.length());
        byte[] buf = new byte[100_003]; // not a block multiple, so reads cross block and chunk boundaries
        long pos = 0;
        while (pos < bytes) {
            int n = (int) Math.min(buf.length, bytes - pos);
            in.readBytes(buf, 0, n);
            for (int i = 0; i < n; i++) {
                if (buf[i] != expected(pos + i)) {
                    fail("wrong byte at " + (pos + i));
                }
            }
            pos += n;
        }
    }

    private long cachedBlocks(String name) {
        sharedCache.getCache().cleanUp();
        Path file = path.resolve(name);
        return sharedCache.getCache().asMap().keySet().stream().filter(k -> ((FileBlockCacheKey) k).filePath().equals(file)).count();
    }

    private long cachedBlocks() {
        sharedCache.getCache().cleanUp();
        return sharedCache.getCache().estimatedSize();
    }

    private void clearCache() {
        sharedCache.getCache().invalidateAll();
        sharedCache.getCache().cleanUp();
    }

    public void testLargeMergeOutputIsNotWriteCached() throws Exception {
        long before = MergeCachePolicy.largeMergeOutputs();
        writeFile("big-merge", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), 4 << 20);
        assertEquals(0, cachedBlocks("big-merge"));
        assertEquals(before + 1, MergeCachePolicy.largeMergeOutputs());
        try (IndexInput in = directory.openInput("big-merge", IOContext.DEFAULT)) {
            assertFileReadsBack(in, 4 << 20);
        }
    }

    public void testSmallMergeOutputIsWriteCached() throws Exception {
        writeFile("small-merge", IOContext.merge(new MergeInfo(1000, largeWriteBytes - 1, false, -1)), 4 << 20);
        assertEquals((4 << 20) / CACHE_BLOCK_SIZE, cachedBlocks("small-merge"));
    }

    public void testFilePastTheLargeWriteSizeIsWriteCachedOnlyUpToIt() throws Exception {
        long bytes = largeWriteBytes + (3 << 20) + 1234; // ends in a partial block, past the limit
        long notCachedBefore = MergeCachePolicy.writeBlocksNotCached();
        writeFile("replicated", IOContext.DEFAULT, bytes);
        assertEquals(largeWriteBytes / CACHE_BLOCK_SIZE, cachedBlocks("replicated"));
        // 384 full blocks past the limit plus the partial tail block.
        assertEquals((3 << 20) / CACHE_BLOCK_SIZE + 1, MergeCachePolicy.writeBlocksNotCached() - notCachedBefore);
        try (IndexInput in = directory.openInput("replicated", IOContext.DEFAULT)) {
            assertFileReadsBack(in, bytes);
        }
    }

    public void testWithBypassOffEverythingIsWriteCached() throws Exception {
        setBypass(false);
        writeFile("big-merge", IOContext.merge(new MergeInfo(1000, largeWriteBytes * 4, false, 1)), 4 << 20);
        assertEquals((4 << 20) / CACHE_BLOCK_SIZE, cachedBlocks("big-merge"));
    }

    public void testReadOnceInputLeavesCacheUnchangedAndReadsCorrectData() throws Exception {
        long bytes = (5 << 20) + 777;
        writeFile("once", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), bytes);
        clearCache();
        long uncachedBefore = MergeCachePolicy.readBytesUncached();
        try (IndexInput in = directory.openInput("once", IOContext.READONCE)) {
            assertFileReadsBack(in, bytes);
            // A slice of a read-once input reads outside the cache too, from any offset.
            IndexInput slice = in.slice("tail", bytes - 20_000, 20_000);
            byte[] tail = new byte[20_000];
            slice.readBytes(tail, 0, tail.length);
            for (int i = 0; i < tail.length; i++) {
                assertEquals(expected(bytes - 20_000 + i), tail[i]);
            }
        }
        assertEquals(0, cachedBlocks());
        assertTrue(MergeCachePolicy.readBytesUncached() - uncachedBefore >= bytes);
    }

    /** Reads the block at {@code offset} through {@code in} and checks it. */
    private static void readBlockAt(IndexInput in, long offset) throws Exception {
        byte[] b = new byte[CACHE_BLOCK_SIZE];
        in.seek(offset);
        in.readBytes(b, 0, b.length);
        for (int i = 0; i < b.length; i++) {
            assertEquals(expected(offset + i), b[i]);
        }
    }

    public void testUncachedRandomReadsReadLittleMoreThanTheirBlocks() throws Exception {
        long bytes = 20 << 20;
        writeFile("random", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), bytes);
        clearCache();
        long readBefore = MergeCachePolicy.readBytesUncached();
        Random rnd = new Random(7);
        int reads = 200;
        try (IndexInput in = directory.openInput("random", IOContext.READONCE)) {
            for (int i = 0; i < reads; i++) {
                readBlockAt(in, (long) rnd.nextInt((int) (bytes / CACHE_BLOCK_SIZE)) * CACHE_BLOCK_SIZE);
            }
            // A clone reads its own blocks the same way.
            IndexInput clone = in.clone();
            readBlockAt(clone, bytes - CACHE_BLOCK_SIZE);
        }
        long read = MergeCachePolicy.readBytesUncached() - readBefore;
        assertTrue("read " + read, read <= (reads + 1L) * CachedByteBufferIndexInput.UNCACHED_MIN_READ_BYTES);
        assertEquals(0, cachedBlocks());
    }

    public void testUncachedSequentialReadGrowsItsReads() throws Exception {
        long bytes = 8 << 20;
        writeFile("seq", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), bytes);
        clearCache();
        long readsBefore = MergeCachePolicy.uncachedReads();
        long bytesBefore = MergeCachePolicy.readBytesUncached();
        try (IndexInput in = directory.openInput("seq", IOContext.READONCE)) {
            assertFileReadsBack(in, bytes);
        }
        long reads = MergeCachePolicy.uncachedReads() - readsBefore;
        // 16, 32, 64, 128 KB, then 256 KB at a time: 8 MB in about 36 reads, each byte read once.
        assertTrue("reads " + reads, reads <= bytes / CachedByteBufferIndexInput.UNCACHED_MAX_READ_BYTES + 6);
        assertEquals(bytes, MergeCachePolicy.readBytesUncached() - bytesBefore);
    }

    public void testUncachedReadsAlternatingBetweenTwoRegionsDontRereadThem() throws Exception {
        long bytes = 16 << 20;
        writeFile("two", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), bytes);
        clearCache();
        long bytesBefore = MergeCachePolicy.readBytesUncached();
        int blocks = 256; // 2 MB from each region
        try (IndexInput in = directory.openInput("two", IOContext.READONCE)) {
            for (int i = 0; i < blocks; i++) {
                readBlockAt(in, (long) i * CACHE_BLOCK_SIZE);
                readBlockAt(in, (8L << 20) + (long) i * CACHE_BLOCK_SIZE);
            }
        }
        long read = MergeCachePolicy.readBytesUncached() - bytesBefore;
        long used = 2L * blocks * CACHE_BLOCK_SIZE;
        assertTrue("read " + read + " for " + used, read <= used + 2L * CachedByteBufferIndexInput.UNCACHED_MAX_READ_BYTES);
    }

    public void testReadOnceInputUsesBlocksAlreadyCached() throws Exception {
        long bytes = 1 << 20;
        writeFile("warm", IOContext.DEFAULT, bytes); // write-cached: every block is in the cache
        assertEquals(bytes / CACHE_BLOCK_SIZE, cachedBlocks("warm"));
        long fromCacheBefore = MergeCachePolicy.readBlocksFromCache();
        long uncachedBefore = MergeCachePolicy.readBytesUncached();
        try (IndexInput in = directory.openInput("warm", IOContext.READONCE)) {
            assertFileReadsBack(in, bytes);
        }
        assertEquals(bytes / CACHE_BLOCK_SIZE, MergeCachePolicy.readBlocksFromCache() - fromCacheBefore);
        assertEquals(uncachedBefore, MergeCachePolicy.readBytesUncached());
    }

    /** Reads the whole file on a new thread with the given name, as an OpenSearch pool thread would. */
    private void readFileOnThread(String threadName, String file, long bytes) throws Exception {
        java.util.concurrent.atomic.AtomicReference<Throwable> failure = new java.util.concurrent.atomic.AtomicReference<>();
        Thread t = new Thread(() -> {
            try (IndexInput in = directory.openInput(file, IOContext.DEFAULT)) {
                assertFileReadsBack(in, bytes);
            } catch (Throwable e) {
                failure.set(e);
            }
        }, threadName);
        t.start();
        t.join();
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
    }

    public void testRecoveryReadsOnTheGenericPoolLeaveCacheUnchanged() throws Exception {
        long bytes = 3 << 20;
        writeFile("recovered", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), bytes);
        clearCache();
        long uncachedBefore = MergeCachePolicy.readBytesUncached();
        readFileOnThread("opensearch[node-1][generic][T#3]", "recovered", bytes);
        assertEquals(0, cachedBlocks());
        assertEquals(bytes, MergeCachePolicy.readBytesUncached() - uncachedBefore);

        // A search thread reading the same file caches it as usual.
        readFileOnThread("opensearch[node-1][search][T#3]", "recovered", bytes);
        assertEquals(bytes / CACHE_BLOCK_SIZE, cachedBlocks("recovered"));

        // With the bypass off, the generic pool caches too.
        clearCache();
        setBypass(false);
        readFileOnThread("opensearch[node-1][generic][T#4]", "recovered", bytes);
        assertEquals(bytes / CACHE_BLOCK_SIZE, cachedBlocks("recovered"));
    }

    public void testDefaultInputStillFillsCache() throws Exception {
        long bytes = 2 << 20;
        writeFile("search", IOContext.merge(new MergeInfo(1000, largeWriteBytes, false, 1)), bytes);
        clearCache();
        try (IndexInput in = directory.openInput("search", IOContext.DEFAULT)) {
            assertFileReadsBack(in, bytes);
        }
        assertEquals(bytes / CACHE_BLOCK_SIZE, cachedBlocks("search"));
    }

    /**
     * A real force merge through IndexWriter's merge threads: with the bypass on, the merge's reads and its
     * large output leave an empty cache empty, and the merged index has every document. With it off, the
     * same merge fills the cache.
     */
    public void testForceMergeLeavesCacheUnchanged() throws Exception {
        assertFalse(MergeCachePolicy.isMergeThread(Thread.currentThread()));

        int common = buildSegments();
        clearCache();
        long uncachedBefore = MergeCachePolicy.readBytesUncached();
        long withBypass = forceMergeAndCheck(common);
        // The merge read its ~24 MB of sources on Lucene's merge threads, outside the cache.
        assertTrue("merge read outside the cache", MergeCachePolicy.readBytesUncached() - uncachedBefore > largeWriteBytes);

        // The same merge with the bypass off, on a fresh set of segments.
        setBypass(false);
        try (IndexWriter writer = new IndexWriter(hybrid, new IndexWriterConfig(new StandardAnalyzer()))) {
            writer.deleteAll();
            writer.commit();
        }
        common = buildSegments();
        clearCache();
        long withoutBypass = forceMergeAndCheck(common);

        logger.info("blocks cached by a force merge: bypass on {}, off {}", withBypass, withoutBypass);
        // Off, the merge's reads and output fill the cache (the write cache keeps only aligned full blocks, so
        // not all of the output); on, it stays empty apart from the few small files a commit writes.
        assertTrue("bypass off cached only " + withoutBypass, withoutBypass > 500);
        assertTrue("bypass on cached " + withBypass + " of " + withoutBypass, withBypass * 20 < withoutBypass);
    }

    /** Writes ~25 MB of documents as 12 segments, returning how many docs have the term "common". */
    private int buildSegments() throws Exception {
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer()).setMergePolicy(new TieredMergePolicy() {
            {
                setSegmentsPerTier(100); // keep the segments apart until the force merge
            }
        });
        Random rnd = new Random(42);
        int common = 0;
        try (IndexWriter writer = new IndexWriter(hybrid, config)) {
            for (int seg = 0; seg < 12; seg++) {
                for (int i = 0; i < 4000; i++) {
                    Document doc = new Document();
                    StringBuilder text = new StringBuilder();
                    for (int w = 0; w < 60; w++) {
                        text.append("w").append(rnd.nextInt(50_000)).append(' ');
                    }
                    boolean isCommon = rnd.nextInt(3) == 0;
                    if (isCommon) {
                        text.append("common");
                        common++;
                    }
                    doc.add(new TextField("body", text.toString(), Field.Store.YES));
                    doc.add(new StringField("id", seg + "-" + i, Field.Store.YES));
                    doc.add(new NumericDocValuesField("n", rnd.nextLong()));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
        }
        return common;
    }

    /** Force merges to one segment, returning the blocks cached right after, then checks the index. */
    private long forceMergeAndCheck(int common) throws Exception {
        try (IndexWriter writer = new IndexWriter(hybrid, new IndexWriterConfig(new StandardAnalyzer()))) {
            writer.forceMerge(1);
            writer.commit();
        }
        long cached = cachedBlocks();
        try (DirectoryReader reader = DirectoryReader.open(hybrid)) {
            assertEquals(1, reader.leaves().size());
            assertEquals(12 * 4000, reader.numDocs());
            assertEquals(common, new IndexSearcher(reader).count(new TermQuery(new Term("body", "common"))));
        }
        return cached;
    }
}
