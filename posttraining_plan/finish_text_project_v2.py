"""Resume the same stage ledger; only S3 launcher command is amended."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from posttraining_plan.rotation_handoff import verify_amendment


def amended_command(command):
    result = list(command)
    if 'posttraining_plan/run_text_formal.py' in result:
        group = result[result.index('--group')+1]
        if group == 'S3':
            result[result.index('posttraining_plan/run_text_formal.py')] = 'posttraining_plan/run_text_formal_v2.py'
    return result


def main():
    verify_amendment()
    from posttraining_plan import finish_text_project as original
    execute = original.execute

    def with_io_amendment(command, log, gpu_ids, timeout):
        verify_amendment()
        return execute(amended_command(command), log, gpu_ids, timeout)

    original.execute = with_io_amendment
    original.main()


if __name__ == '__main__':
    main()
