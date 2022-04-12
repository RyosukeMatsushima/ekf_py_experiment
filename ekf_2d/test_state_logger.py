from unittest import TestCase
from state_logger import StateLogger

class TestStateLogger(TestCase):

    def setUp(self):
        self.stateLogger = StateLogger('state_logger_test.csv', ('a', 'b'))

    def test_logging(self):
        for i in range(100001):
            self.stateLogger.add_data([2 * i, 3 * i])

#TODO: add tearDown() and remove state_logger_test.csv

