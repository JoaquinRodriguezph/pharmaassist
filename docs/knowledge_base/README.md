# Knowledge base

Four fictional documents used to test the PharmaAssist RAG pipeline. Each starts with the banner *"FICTIONAL document for software testing only. Not medical or pharmacy guidance."* This README is not a knowledge-base document: the chunker must **not** index it.

| File | Sections cited by `eval/test_set.json` |
|---|---|
| `returns_and_expiry_policy.md` | Customer Returns, Expired Medications |
| `storage_sop.md` | Temperature Monitoring, Cold Chain And Excursions, Receiving And FEFO |
| `paracetamol_leaflet.md` | Dosage And Administration, Warnings And Precautions |
| `ibuprofen_leaflet.md` | Contraindications, Drug Interactions |

Each file also has one uncited section ("Purpose And Scope" or "About This Medicine"), so the chunker and retriever meet sections that no question points at.

## Heading rule

**Headings are written as readable text. Code turns them into the hyphenated form used in citations.** The test set cites `storage_sop.md#Cold-Chain-And-Excursions`; the heading in the file is `## Cold Chain And Excursions`.

1. A **section** is a level-2 heading: a line starting with exactly `## `. The `#` line is the document title. `###` lines are subsections and belong to the section above them. Headings are never written inside code blocks.
2. The **fragment** is `slugify_heading(heading)` from `app/rag/kb.py`: drop every character that is not a letter, digit, underscore, whitespace or hyphen; turn each run of whitespace or hyphens into one hyphen; trim hyphens from the ends. **Capitalization is kept.**
3. A **section ID** is `<file>.md#<fragment>`. Matching against citations is **exact and case-sensitive**.
4. Because capitalization is kept, headings are written so that their fragment equals the test set exactly. That is why "And" is capitalized and the acronym stays uppercase: `Receiving And FEFO` becomes `Receiving-And-FEFO`.
5. Fragments must be unique within a file.

| Heading in the file | Fragment |
|---|---|
| `## Customer Returns` | `Customer-Returns` |
| `## Cold Chain And Excursions` | `Cold-Chain-And-Excursions` |
| `## Receiving And FEFO` | `Receiving-And-FEFO` |

The chunker (to build section IDs) and the citation checker (to compare them) must both call `slugify_heading`, never their own version of the rule.

## Writing a section

Each section is 80 to 200 words, in the voice of a real SOP or product leaflet, and carries the facts the matching `expected_answer` ("Spec facts for Step 3") asks for.

## Deliberate absences

These topics must stay out of every knowledge-base file, because the test set checks that the assistant admits it does not know, or refuses:

- **Q09 and Q10:** no employee-discount policy, and no offline-POS or e-wallet payment procedure.
- **Q20:** no children's or weight-based dosing numbers in the paracetamol leaflet. It states that the leaflet covers adult 500 mg tablets only and that children's dosing must be calculated by a pharmacist or physician.

If a question is changed so that it should be answerable, change the knowledge base and the test set together.

## Required strings (`must_contain`)

Each RAG question in `eval/test_set.json` (Q01 to Q08, Q17, Q18, Q19) has a `must_contain` list: short strings taken from its `pass_criteria` that the cited section must contain.

- **List form:** every section in the question's `expected_sources` must contain all the strings.
- **Object form:** used when a question cites several sections that do not share the same facts (Q18). Each key is one of the question's `expected_sources`, and its value is the list for that section.
- **Where it looks:** the section's text, from the line after its `##` heading to the next `##` heading. The heading line itself is not searched, so a word that appears only in a heading (such as "FEFO" in `Receiving And FEFO`) does not count.
- **How it matches:** plain text, ignoring case. Any run of whitespace, including a line break, counts as one space. Nothing else is normalized, so `8 tablets` and `eight tablets` are different strings, and so are `2°C–8°C` (with an en dash) and `2-8°C`.
- **One wording per fact:** the documents use the same wording everywhere a fact appears: `15°C–25°C`, `2°C–8°C`, `8:00 AM` and `4:00 PM`, `30 minutes`, `7 calendar days`, `8 tablets`, `4000 mg`. If you reword a fact in a document, change the string in the test set to match; never shorten a string just to make the check pass.

## Honest limits

- **These documents were written after the questions were seen.** Each section was written to contain exactly the facts its question asks for, in the questions' own terms. Retrieval and answer scores on this test set will therefore look better than they would on real, unseen documents and questions. Treat them as a check that the pipeline works, not as a measure of how well it will do in use.
- **String matching proves a fact is present in the text. It does not prove a model's answer is correct.** `must_contain` can be reused as a cheap automated check on answers, but it misses paraphrases, such as "three months" for `90 days`. Read a sample of answers by hand as well.

## Checking

```
py scripts/check_kb.py
```

Reads `eval/test_set.json`, collects every `expected_sources` entry and every `must_contain` list, and checks that each file exists, has a heading whose fragment matches, and that the section's text contains every required string. It prints each problem, for example `MISSING: storage_sop.md#Temperature-Monitoring (section text is missing '8:00 AM')`, and exits with status 1 if there are any. A malformed `must_contain` also exits with status 1.