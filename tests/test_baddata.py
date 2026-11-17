import unittest

from VISSSlib import tools


def _config():
    return tools.DictNoDefault(
        {
            "badData": [
                {
                    "start": "20250812-190000",
                    "end": "20250911-090000",
                    "reason": "missed frames",
                    "products": ["level1match", "level2detect"],
                }
            ]
        }
    )


class TestBadPeriod(unittest.TestCase):
    def test_daily_level_whole_days(self):
        c = _config()
        self.assertTrue(tools.isBadPeriod("20250812", c, product="level2detect")[0])
        self.assertTrue(tools.isBadPeriod("20250911", c, product="level2detect")[0])
        self.assertFalse(tools.isBadPeriod("20250811", c, product="level2detect")[0])
        # the day after the end date is not affected (see isBadPeriod)
        self.assertFalse(tools.isBadPeriod("20250912", c, product="level2detect")[0])

    def test_file_level_exact_times(self):
        c = _config()
        self.assertFalse(
            tools.isBadPeriod("20250812-185000", c, product="level1match")[0]
        )
        self.assertTrue(
            tools.isBadPeriod("20250812-190000", c, product="level1match")[0]
        )
        self.assertTrue(
            tools.isBadPeriod("20250911-090000", c, product="level1match")[0]
        )
        self.assertFalse(
            tools.isBadPeriod("20250911-091000", c, product="level1match")[0]
        )

    def test_product_selection(self):
        c = _config()
        # not listed: level1detect is kept
        self.assertFalse(
            tools.isBadPeriod("20250815-120000", c, product="level1detect")[0]
        )
        # camera prefixed names are matched on the level
        self.assertTrue(
            tools.isBadPeriod("20250815-120000", c, product="leader_level1match")[0]
        )
        self.assertEqual(
            tools.isBadPeriod("20250815", c, product="level2detect")[1], "missed frames"
        )


    def test_file_level_date_only_period(self):
        # date-only start/end used to crash per-file levels (ljust padding
        # produced "202606280000000"); the whole end day is included
        c = tools.DictNoDefault(
            {
                "badData": [
                    {
                        "start": "20260628",
                        "end": "20260629",
                        "reason": "r",
                        "products": None,
                    }
                ]
            }
        )
        for case, bad in [
            ("20260627-235000", False),
            ("20260628-000000", True),
            ("20260629-235000", True),
            ("20260630-000000", False),
        ]:
            self.assertEqual(
                tools.isBadPeriod(case, c, product="leader_level1detect")[0], bad, case
            )
