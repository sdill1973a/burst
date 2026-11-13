package com.gtc.burst

import org.junit.Assert.*
import org.junit.Test

class Wire2VectorsTest {
    private val rows = javaClass.classLoader!!.getResourceAsStream("wire2_vectors.txt")!!.bufferedReader().readLines().filter { it.isNotBlank() }.map { it.split("\t") }
    private fun un(s: String) = Wire2.unhex(s)
    private fun str(s: String) = String(un(s), Charsets.UTF_8)
    private fun list(s: String) = if (s == "-") emptyList() else s.split(",").map { str(it) }
    private val prime = str(rows.first { it[0] == "prime" }[1])
    private fun of(k: String) = rows.filter { it[0] == k }
    private val texts = of("map").associate { it[1] to str(it[2]) }
    private fun decode(f: ByteArray, @Suppress("UNUSED_PARAMETER") t: List<List<String>>) =
        texts[Wire2.hex(f)]?.let { Codec.Decoded(it, 0, true, 8) } ?: Codec.Decoded("", 0, false, 8, "refused")
    private fun asm() = Chain.Assembler2(::decode)
    private fun frames(k: String) = of(k).single()[2].split(",").map { un(it) }

    @Test fun constantOn() = assertTrue(Wire2.SEND)
    @Test fun tailIdsAndPrimes() { assertTrue(of("tail").size >= 6); for (r in of("tail")) {
        val t = list(r[1]).takeLast(Codec.TAIL_N); val p = Wire2.primeFor(prime, t)
        assertEquals(r[3], Wire2.hex(Wire2.sha(p.toByteArray()))); assertEquals(r[2].toInt(), Wire2.tailId(p)) } }
    @Test fun framesProducedAndAccepted() { assertTrue(of("frame").size >= 4); for (r in of("frame")) {
        val text = un(r[1]); val carried = r[3] == "1"; val old = un(r[5]); val new = un(r[6])
        assertArrayEquals(new, Wire2.fromDet08(old, r[4].toInt(), carried, text)); assertArrayEquals(old, Wire2.toDet08(new))
        assertEquals(if (carried) Wire2.CARRY else Wire2.CODEC, new[0]); assertArrayEquals(Wire2.check3(text), new.copyOfRange(2, 5)) } }
    @Test fun askBytes() { assertTrue(of("ask").isNotEmpty()); for (r in of("ask")) assertEquals(r[2], Wire2.hex(Wire2.askFrame(un(r[1])))) }
    @Test fun linksProducedAndAccepted() { assertTrue(of("link").size >= 16); for (r in of("link")) {
        val whole = if (r[5] == "-") null else un(r[5])
        val b = Wire2.link(r[1].toInt(), r[2].toInt(), r[3].toInt(), un(r[4]), whole); assertEquals(r[6], Wire2.hex(b))
        val p = Wire2.parseLink(b)!!; assertEquals(r[1].toInt(), p.mid); assertEquals(r[2].toInt(), p.idx); assertEquals(r[3].toInt(), p.total)
        assertArrayEquals(un(r[4]), p.inner); if (whole == null) assertNull(p.whole) else assertArrayEquals(whole, p.whole) } }
    @Test fun carryMatches() { for (r in of("carry")) assertEquals(list(r[3]), Chain.carry(list(r[1]), str(r[2]))) }
    @Test fun honestChainJoins() { val a = asm(); var s: Chain.Step = Chain.Step.Waiting
        for (f in frames("chainA")) s = a.add(1L, f, emptyList(), 1000L)
        assertEquals(str(of("chainA").single()[1]), (s as Chain.Step.Done).whole.text) }
    @Test fun spliceRefused() { val a = asm(); val fa = frames("chainA"); val fb = frames("chainB"); var s: Chain.Step = Chain.Step.Waiting
        for (i in fa.indices) s = a.add(2L, if (i < 4) fa[i] else fb[i], emptyList(), 1000L)
        assertTrue(s is Chain.Step.Refused) ; println("splice: " + (s as Chain.Step.Refused).why) }
    @Test fun droppedLinkRecovers() { val a = asm(); val fa = frames("chainA"); val r = of("missing").single()
        for (i in fa.indices) if (i != r[2].toInt().countTrailingZeroBits()) assertTrue(a.add(3L, fa[i], emptyList(), 1000L) is Chain.Step.Waiting)
        assertTrue(a.due(1000L + 44_000L).isEmpty())
        val n = a.due(1000L + 46_000L).single(); assertEquals(r[3], Wire2.hex(n.frame!!)); assertEquals(3L, n.from)
        val s = a.add(3L, fa[r[2].toInt().countTrailingZeroBits()], emptyList(), 60_000L)
        assertEquals(str(of("chainA").single()[1]), (s as Chain.Step.Done).whole.text); assertTrue(a.idle) }
    @Test fun refusedLinkAsksWithItsKey() { val fa = frames("chainA"); val d = { f: ByteArray, t: List<List<String>> -> if (Wire2.hex(f) == Wire2.hex(Wire2.parseLink(fa[2])!!.inner)) Codec.Decoded("", 0, false, 8, "no") else decode(f, t) }
        val a = Chain.Assembler2(d); var s: Chain.Step = Chain.Step.Waiting; for (f in fa) s = a.add(4L, f, emptyList(), 1000L)
        assertEquals(Wire2.hex(Wire2.askFrame(fa[2])), Wire2.hex((s as Chain.Step.Ask).frames.single())) }
    @Test fun givesUpAfterThreeAsks() { val a = asm(); a.add(5L, frames("chainA")[0], emptyList(), 0L); var t = 0L; var asks = 0
        repeat(3) { i -> t += 45_000L * (i + 1); if (a.due(t).single().frame != null) asks++ }; t += 45_000L * 4
        assertNull(a.due(t).single().frame); assertEquals(3, asks); assertTrue(a.idle) }
}
