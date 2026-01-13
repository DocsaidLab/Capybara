from capybara.utils.custom_tqdm import Tqdm


def test_custom_tqdm_respects_explicit_total_and_infers_total():
    bar = Tqdm(range(3), total=10, disable=True)
    try:
        assert bar.total == 10
    finally:
        bar.close()

    inferred = Tqdm([1, 2, 3], disable=True)
    try:
        assert inferred.total == 3
    finally:
        inferred.close()
