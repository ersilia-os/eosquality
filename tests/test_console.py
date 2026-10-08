def test_plural_follows_the_count():
    from eosquality.utils import console

    assert [console.plural(n, "column") for n in (0, 1, 2)] == [
        "0 columns",
        "1 column",
        "2 columns",
    ]
