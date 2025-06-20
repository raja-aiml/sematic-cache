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
DETAILED=false

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
        -d|--detailed)
            DETAILED=true
            COVERAGE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -q, --quiet     Run tests without verbose output"
            echo "  -c, --coverage  Generate coverage report"
            echo "  -d, --detailed  Show detailed per-package coverage"
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
        if [ "$DETAILED" = true ]; then
            echo -e "${YELLOW}Calculating detailed coverage...${NC}"
            
            # Initialize totals
            total_statements=0
            covered_statements=0
            
            # Run tests for each package and extract coverage
            for pkg in config pkg/utils pkg/docker pkg/database pkg/secrets pkg/constants pkg/testing pkg/k3d pkg/kubernetes pkg/cmd cmd internal/build internal/build/docker; do
                if [ -d "$pkg" ]; then
                    echo -e "${YELLOW}Testing $pkg...${NC}"
                    output=$(go test -cover ./$pkg 2>&1)
                    
                    # Extract coverage percentage if tests exist
                    coverage=$(echo "$output" | grep -E "coverage: [0-9]+\.[0-9]% of statements" | grep -oE "[0-9]+\.[0-9]")
                    
                    if [ ! -z "$coverage" ]; then
                        # Get number of statements by running coverage analysis
                        go test -coverprofile=temp.out ./$pkg 2>/dev/null
                        if [ -f "temp.out" ]; then
                            statements=$(go tool cover -func=temp.out 2>/dev/null | grep "total:" | awk '{print $2}' | sed 's/[^0-9]//g')
                            if [ ! -z "$statements" ]; then
                                covered=$(echo "scale=0; $statements * $coverage / 100" | bc 2>/dev/null || echo "0")
                                total_statements=$((total_statements + statements))
                                covered_statements=$((covered_statements + covered))
                                echo -e "  Coverage: ${GREEN}$coverage%${NC} ($covered/$statements statements)"
                            fi
                            rm -f temp.out
                        fi
                    fi
                fi
            done
            
            # Calculate overall coverage
            if [ $total_statements -gt 0 ]; then
                overall_coverage=$(echo "scale=2; $covered_statements * 100 / $total_statements" | bc 2>/dev/null || echo "0")
                echo ""
                echo -e "${GREEN}Overall Coverage: $overall_coverage% ($covered_statements/$total_statements statements)${NC}"
            fi
        fi
        
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