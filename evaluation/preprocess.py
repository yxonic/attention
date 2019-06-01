"""Data preprocessing module. All image/label data should be processed into the
following structure:

```
data/
  dataset/
    imgs/
      id1.png
      id2.png
      ...
    label.txt
```

Inside `label.txt`, each line should be tab separated, and should contain two
field: `image_id` and `label` (space separated words/tokens/characters).
"""

import logging
import shutil
from pathlib import Path
import fret


@fret.command
def prep_formula(data_dir='data/formula', replace=False):
    data_dir = Path(data_dir)
    if data_dir.exists() and not replace:
        logging.info(f'directory {data_dir} already exists')
    elif data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    # TODO: process data into `data_dir`
    logging.error('not processed')
