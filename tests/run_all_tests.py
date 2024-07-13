import unittest

# Import all the test modules
import tests.test_assistant_manager
import tests.test_file_manager
import tests.test_openai_assistant
import tests.test_thread_manager
import tests.test_tool_functions_map
import tests.test_utils
import tests.test_vector_stores_manager


def main():
    # Create a TestSuite
    suite = unittest.TestSuite()

    # Add tests from each module
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_assistant_manager))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_file_manager))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_openai_assistant))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_thread_manager))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_tool_functions_map))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_utils))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(tests.test_vector_stores_manager))

    # Run the tests and display the results
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Display detailed error information
    if not result.wasSuccessful():
        if result.errors:
            print("\nErrors:")
            error_files = set(error[0].id().split('.')[1] for error in result.errors)
            for file in error_files:
                print(f"In file: {file}")
            for error_test, reason in result.errors:
                print(f"{error_test}: {reason}")

        if result.failures:
            print("\nFailures:")
            for failed_test, reason in result.failures:
                print(f"{failed_test}: {reason}")

    # Print a summary
    if result.errors or result.failures:
        print("\n-----------------------\nERROR LIST:")
        for error in result.errors:
            print(f"Error in {error[0].id()}")
        print("-----------------------\n\n")
    print("Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.wasSuccessful():
        print("All tests passed successfully!")


if __name__ == '__main__':
    main()
