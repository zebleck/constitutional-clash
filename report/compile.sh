#!/bin/bash

# Constitutional Clash LaTeX Compilation Script
# Compiles the research paper with proper bibliography and cross-references

set -e  # Exit on any error

echo "=== Constitutional Clash Paper Compilation ==="
echo "Starting LaTeX compilation process..."

# Check if required files exist
if [ ! -f "main.tex" ]; then
    echo "Error: main.tex not found!"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo "Warning: references.bib not found! Bibliography may not work properly."
fi

# First pass - generate aux files
echo "First pass: Generating auxiliary files..."
pdflatex -interaction=nonstopmode main.tex

# Check if biber/bibtex is needed
if [ -f "references.bib" ] && grep -q "\\\\cite\|\\\\biblatex" main.tex; then
    echo "Second pass: Processing bibliography..."
    biber main 2>/dev/null || bibtex main 2>/dev/null || echo "Warning: Bibliography processing failed"
fi

# Second pass - resolve cross-references
echo "Third pass: Resolving cross-references..."
pdflatex -interaction=nonstopmode main.tex

# Third pass - final cleanup
echo "Fourth pass: Final compilation..."
pdflatex -interaction=nonstopmode main.tex

# Check if PDF was created successfully
if [ -f "main.pdf" ]; then
    echo "✅ SUCCESS: main.pdf generated successfully!"
    echo "📄 Output: $(pwd)/main.pdf"
    
    # Show file size
    size=$(stat -c%s main.pdf 2>/dev/null || stat -f%z main.pdf 2>/dev/null || echo "unknown")
    if [ "$size" != "unknown" ]; then
        echo "📊 File size: $(numfmt --to=iec --suffix=B $size)"
    fi
    
    # Count pages if pdfinfo is available
    if command -v pdfinfo >/dev/null 2>&1; then
        pages=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [ -n "$pages" ]; then
            echo "📖 Total pages: $pages"
        fi
    fi
    
    echo ""
    echo "🎉 Constitutional Clash research paper compiled successfully!"
    echo "   Ready for academic submission or presentation."
    
else
    echo "❌ ERROR: PDF compilation failed!"
    echo "Check the LaTeX log file (main.log) for error details."
    exit 1
fi

# Cleanup auxiliary files (optional - uncomment if desired)
# echo "Cleaning up auxiliary files..."
# rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc

echo "=== Compilation Complete ==="