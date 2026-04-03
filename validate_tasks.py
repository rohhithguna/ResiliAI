from src.inference.inference import run_task

from tasks.easy_task import get_easy_task
from tasks.medium_task import get_medium_task
from tasks.hard_task import get_hard_task

from graders.easy_grader import grade_easy
from graders.medium_grader import grade_medium
from graders.hard_grader import grade_hard


PASS_THRESHOLD = 0.65


def _validate_single(label, task_getter, grader):
    task = task_getter()
    result = run_task(task)
    grade = float(grader(result))
    passed = grade >= PASS_THRESHOLD
    status = "PASS" if passed else "FAIL"
    print(f"{label}: {status}")
    return passed, result, grade


def main():
    outcomes = []

    outcomes.append(_validate_single("EASY", get_easy_task, grade_easy))
    outcomes.append(_validate_single("MEDIUM", get_medium_task, grade_medium))
    outcomes.append(_validate_single("HARD", get_hard_task, grade_hard))

    all_pass = all(item[0] for item in outcomes)
    if all_pass:
        print("FINAL: PASS ✅")
        return 0

    print("FINAL: FAIL ❌")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
