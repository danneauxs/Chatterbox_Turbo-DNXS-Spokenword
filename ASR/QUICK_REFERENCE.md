# ASR Validator - Quick Reference Card

## What Changed

The validator now uses **dual-channel normalization** and **hybrid scoring** to handle prose and IDs separately.

## Key Files

| File | Purpose |
|------|---------|
| `asr_validator.py` | Main application (modified) |
| `CHANGES.md` | Summary of all changes |
| `IMPLEMENTATION_SUMMARY.md` | Detailed technical documentation |
| `test_normalization.py` | Standalone test script |
| `AGENTS.md` | Updated developer guidelines |

## Testing the Changes

### Quick Test
```bash
# Run the normalization test
python3 test_normalization.py
```

### Full Test
```bash
# Launch the GUI
python3 asr_validator.py

# Or use the launcher
./run.sh
```

## How It Works Now

### 1. Normalization (Dual Channel)

**Before:**
```python
ref_normalized = normalize("R-KK1418991 is not responding")
# Result: "r kk one million four hundred eighteen thousand..."
```

**After:**
```python
ref_normalized, ref_ids = normalize("R-KK1418991 is not responding")
# Result: ("<ID0> is not responding", ["rkk1418991"])
```

### 2. Scoring (Hybrid)

**Before:**
```python
score = similarity(ref_normalized, hyp_normalized)
# Single score (0-1)
```

**After:**
```python
prose_score = similarity(ref_prose, hyp_prose)
id_score = len(ref_ids & hyp_ids) / len(ref_ids)
combined_score = 0.7 * prose_score + 0.3 * id_score
# Separate + combined scoring
```

### 3. Pass/Fail (Strict)

**Before:**
```python
passed = score >= threshold
# Hallucinations ignored if score is high
```

**After:**
```python
passed = (
    combined_score >= threshold and
    not is_truncated and
    not is_hallucinated
)
# Strict enforcement
```

## Expected Results

### Chunk 02467 (ID Mismatch)
- **Before:** Score 0.50 - confusing "missing/extra" list
- **After:** Clear ID diagnosis - "missing IDs: rkk1418991; extra IDs: rkk148991"

### Chunk 02684 (Repetition)
- **Before:** Score 0.75 - warning but might pass
- **After:** Hard failure - "Hallucination detected: 'Hope!' repeated 4 times"

### Chunk 03813 (Coherent Hallucination)
- **Before:** Score 0.68 - set-based diff confusing
- **After:** Hard failure - "extra: and well talk more about that..."

## Configuration

### Tuning Scoring Weights
Edit `asr_validator.py` around line 700:
```python
combined_score = 0.7 * prose_score + 0.3 * id_score
```

**Recommendations:**
- **ID-heavy content:** Use 0.5/0.5 or 0.4/0.6
- **Prose-only content:** Use 0.9/0.1
- **Balanced (default):** Use 0.7/0.3

### Tuning Repetition Threshold
Edit `asr_validator.py` around line 255:
```python
if count >= 3:  # Collapse threshold
```

**Recommendations:**
- **Conservative:** Use 4 or 5
- **Aggressive:** Use 2
- **Balanced (default):** Use 3

## Troubleshooting

### "Syntax error" when running
```bash
# Check Python version (need 3.8+)
python3 --version

# Recompile
python3 -m py_compile asr_validator.py
```

### "Import error" for torch/librosa
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or reinstall
pip install -r requirements.txt
```

### Unexpected failures after update
1. Check validation.log for new diagnostic fields
2. Review `prose_score` vs `id_score` breakdown
3. Check for `missing_ids` / `extra_ids` fields
4. Adjust scoring weights if needed

## Rollback (If Needed)

If you need to revert the changes:
```bash
# Restore from git (if tracked)
git checkout HEAD -- asr_validator.py

# Or restore from backup
cp asr_validator.py.backup asr_validator.py
```

## Support Files

- **CHANGES.md** - Quick summary of what changed
- **IMPLEMENTATION_SUMMARY.md** - Full technical details
- **AGENTS.md** - Developer guidelines (updated)
- **test_normalization.py** - Unit tests

## Next Steps

1. ✅ Test with `test_normalization.py`
2. ✅ Verify GUI launches: `python3 asr_validator.py`
3. 🔲 Run validation on production TTS output
4. 🔲 Review new diagnostic fields in logs
5. 🔲 Adjust weights if needed
6. 🔲 Monitor for edge cases

---

**Questions?** Review `IMPLEMENTATION_SUMMARY.md` for detailed explanations.
