"""Atomic JSON writes with retries for Windows reader/antivirus file locks."""
import json
import os
from pathlib import Path
import tempfile
import time
import warnings


def write_json(path, value, *, best_effort=False, attempts=12):
    path = Path(path)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(.01 * 2**attempt, .2))
    except OSError as exc:
        if not best_effort:
            raise
        warnings.warn(f'Display update skipped for {path.name}: {exc}', RuntimeWarning)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
