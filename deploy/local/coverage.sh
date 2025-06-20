#!/bin/bash

# Run tests and collect coverage
echo "Running tests with coverage..."

# Initialize totals
total_statements=0
covered_statements=0

# Run tests for each package and extract coverage
for pkg in config pkg/utils pkg/docker pkg/database pkg/secrets pkg/constants pkg/testing pkg/k3d pkg/kubernetes pkg/cmd; do
    if [ -d "$pkg" ]; then
        echo "Testing $pkg..."
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
                    echo "  Coverage: $coverage% ($covered/$statements statements)"
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
    echo "Overall Coverage: $overall_coverage% ($covered_statements/$total_statements statements)"
else
    echo "No testable statements found"
fi