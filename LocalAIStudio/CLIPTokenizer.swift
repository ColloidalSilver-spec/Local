import Foundation

/// Minimal CLIP BPE tokenizer — a direct port of the web app's tokenizer
/// (transformers.js `PreTrainedTokenizer` built from vocab.json + merges.txt by
/// `buildClipTokenizerJson` in src/image.js). Same normalization, same
/// pre-tokenization, same BPE merge ranking, same BOS/EOS/padding — so it
/// produces the same token ids as the verified web-side pipeline.
final class CLIPTokenizer {
    let bosId: Int
    let eosId: Int
    let unkId: Int
    let maxLen: Int
    private let vocab: [String: Int]
    private let mergeRanks: [String: Int]

    init(vocabJSON: Data, mergesText: String, bosId: Int = 49406, eosId: Int = 49407, maxLen: Int = 77) {
        self.bosId = bosId
        self.eosId = eosId
        // CLIP has no <unk> token; transformers falls back to <|endoftext|> for
        // characters the BPE can't represent.
        self.unkId = eosId
        self.maxLen = maxLen

        var v: [String: Int] = [:]
        if let obj = try? JSONSerialization.jsonObject(with: vocabJSON) as? [String: Any] {
            for (k, val) in obj {
                if let i = val as? Int { v[k] = i }
            }
        }
        self.vocab = v

        var ranks: [String: Int] = [:]
        var idx = 0
        for line in mergesText.components(separatedBy: .newlines) {
            let t = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if t.isEmpty || t.hasPrefix("#") { continue }
            let parts = t.split(separator: " ", maxSplits: 1).map(String.init)
            if parts.count == 2 {
                ranks[parts[0] + "\u{1}" + parts[1]] = idx
                idx += 1
            }
        }
        self.mergeRanks = ranks
    }

    // BertNormalizer: clean_text (strip control chars, collapse whitespace),
    // lowercase, handle_chinese_chars (pad CJK with spaces).
    private func normalize(_ text: String) -> String {
        var out = ""
        var prevSpace = false
        var prev: Character = " "
        for scalar in text.lowercased().unicodeScalars {
            let ch = Character(scalar)
            // strip C0 control chars except tab/newline/cr
            if scalar.value < 32 && scalar.value != 9 && scalar.value != 10 && scalar.value != 13 { continue }
            let isSpace = scalar.value == 0x20 || scalar.value == 0x09 || scalar.value == 0x0A || scalar.value == 0x0D
                || (scalar.value >= 0x2000 && scalar.value <= 0x200B) || scalar.value == 0x00A0
            if isSpace {
                if !prevSpace { out.append(" "); prev = " " }
                prevSpace = true
                continue
            }
            prevSpace = false
            let cjk = (scalar.value >= 0x4E00 && scalar.value <= 0x9FFF)
                || (scalar.value >= 0x3400 && scalar.value <= 0x4DBF)
                || (scalar.value >= 0xF900 && scalar.value <= 0xFAFF)
                || (scalar.value >= 0x3040 && scalar.value <= 0x30FF)
                || (scalar.value >= 0xAC00 && scalar.value <= 0xD7AF)
            if cjk {
                if prev != " " { out.append(" ") }
                out.append(ch)
                out.append(" ")
                prev = " "
            } else {
                out.append(ch)
                prev = ch
            }
        }
        return out
    }

    // BertPreTokenizer: split on whitespace and the punctuation set (),.!?;:'"
    // (punctuation pieces become their own words, matching the tokenizers lib).
    private func preTokenize(_ text: String) -> [String] {
        let punct: Set<Character> = ["(", ")", ",", ".", "!", "?", ";", ":", "'", "\""]
        var words: [String] = []
        var cur = ""
        for ch in text {
            if ch == " " {
                if !cur.isEmpty { words.append(cur); cur = "" }
            } else if punct.contains(ch) {
                if !cur.isEmpty { words.append(cur); cur = "" }
                words.append(String(ch))
            } else {
                cur.append(ch)
            }
        }
        if !cur.isEmpty { words.append(cur) }
        return words
    }

    // BPE: character-level tokens (last char gets </w>), then repeatedly merge
    // the adjacent pair with the lowest merge rank until none remains.
    private func bpe(_ word: String) -> [String] {
        var chars = Array(word).map { String($0) }
        if !chars.isEmpty { chars[chars.count - 1] += "</w>" }
        var toks = chars
        while toks.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0 ..< (toks.count - 1) {
                if let r = mergeRanks[toks[i] + "\u{1}" + toks[i + 1]] {
                    if r < bestRank { bestRank = r; bestIdx = i; if r == 0 { break } }
                }
            }
            if bestIdx < 0 { break }
            toks[bestIdx] = toks[bestIdx] + toks[bestIdx + 1]
            toks.remove(at: bestIdx + 1)
        }
        return toks
    }

    func encode(_ text: String) -> [Int32] {
        let normalized = normalize(text)
        var ids: [Int32] = [Int32(bosId)]
        for w in preTokenize(normalized) {
            for tok in bpe(w) {
                ids.append(Int32(vocab[tok] ?? unkId))
            }
        }
        ids.append(Int32(eosId))
        if ids.count > maxLen {
            var t = [Int32(bosId)]
            t.append(contentsOf: ids[1 ..< (1 + (maxLen - 2))])
            t.append(Int32(eosId))
            ids = t
        }
        while ids.count < maxLen { ids.append(Int32(eosId)) }
        return ids
    }

    func describe(ids: [Int32]) -> [String] {
        var rev: [Int: String] = [:]
        for (k, v) in vocab { rev[v] = k }
        return ids.map { rev[Int($0)] ?? "?" }
    }
}
