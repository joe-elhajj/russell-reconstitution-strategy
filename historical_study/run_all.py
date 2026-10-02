#!/usr/bin/env python
"""
Master runner to execute the pipeline scripts in order.
"""
import time
from pathlib import Path

SCRIPTS = [
    ('STEP 1/4: 26-YEAR BACKTEST', 'scripts.backtest_full'),
    ('STEP 2/4: Per-ticker models', 'scripts.per_ticker_models'),
    ('STEP 3/4: Monte Carlo sizing', 'scripts.monte_carlo'),
    ('STEP 4/4: December recon preview', 'scripts.december_recon'),
]

results = []
outputs = []

for header, module_name in SCRIPTS:
    print('\n' + '='*60)
    print(header)
    print('='*60)
    t0 = time.time()
    try:
        module = __import__(module_name, fromlist=['main'])
        if hasattr(module, 'main'):
            module.main()
            status = 'success'
            outputs.append(module_name)
        else:
            status = 'no_main'
    except Exception as e:
        status = f'failed: {e}'
    t1 = time.time()
    results.append({'script': module_name, 'status': status, 'runtime_s': t1 - t0})

# Summary
print('\nPROCESS SUMMARY')
for r in results:
    print(f"{r['script']}: {r['status']} (runtime {r['runtime_s']:.1f}s)")

from pathlib import Path
result_files = list(Path('results').glob('*'))
print('\nResult files:')
for f in result_files:
    print('-', f)

print('\nRun complete')
