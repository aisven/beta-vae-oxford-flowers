#!/bin/bash
CONTENTDIR="content"
BUILDDIR="target"
FILENAME="document"
ASSETSDIR="assets"

download_csl() {
    wget -O "${ASSETSDIR}/harvard-anglia-ruskin-university.csl" \
        "https://raw.githubusercontent.com/citation-style-language/styles/master/harvard-anglia-ruskin-university.csl"
}

pdf_print() {
    mkdir -p "${BUILDDIR}"
    echo "Creating pdf-print output"
    pandoc "${CONTENTDIR}/${FILENAME}.md" \
        --resource-path="${CONTENTDIR}" \
        --citeproc \
        --csl="${ASSETSDIR}/harvard-anglia-ruskin-university-custom.csl" \
        --from="markdown+tex_math_single_backslash+tex_math_dollars+raw_tex" \
        --to="latex" \
        --output="${BUILDDIR}/output_print.pdf" \
        --pdf-engine="xelatex" \
        --include-in-header="layouts/print.tex"
}

pdf_ereader() {
    mkdir -p "${BUILDDIR}"
    echo "Creating pdf-ereader output"
    pandoc "${CONTENTDIR}/${FILENAME}.md" \
        --resource-path="${CONTENTDIR}" \
        --citeproc \
        --csl="${ASSETSDIR}/harvard-anglia-ruskin-university-custom.csl" \
        --from="markdown+tex_math_single_backslash+tex_math_dollars+raw_tex" \
        --to="latex" \
        --output="${BUILDDIR}/output_ereader.pdf" \
        --pdf-engine="xelatex" \
        --include-in-header="layouts/ereader.tex"
}

# Allows to call a function based on arguments passed to the script
# Example: `./generate_document.sh pdf_print`
$*
