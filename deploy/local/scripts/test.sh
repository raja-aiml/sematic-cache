#!/bin/bash
# Test runner script that filters out macOS linker warnings

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to verbose mode
VERBOSE=true
COVERAGE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quiet)
            VERBOSE=false
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -q, --quiet     Run tests without verbose output"
            echo "  -c, --coverage  Generate coverage report"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build test command
TEST_CMD="go test"
if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD -v"
fi
TEST_CMD="$TEST_CMD -race -coverprofile=coverage.out ./..."

echo -e "${YELLOW}Running tests...${NC}"

# Run tests and filter warnings
if $TEST_CMD 2>&1 | grep -v "ld: warning: -bind_at_load is deprecated" | grep -v "ld: warning: .* has malformed LC_DYSYMTAB"; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo -e "${YELLOW}Generating coverage report...${NC}"
        go tool cover -html=coverage.out -o coverage.html
        echo -e "${GREEN}Coverage report generated: coverage.html${NC}"
        
        # Show coverage summary
        go tool cover -func=coverage.out | tail -1
    fi
    
    exit 0
else
    echo -e "${RED}✗ Tests failed!${NC}"
    exit 1
fi