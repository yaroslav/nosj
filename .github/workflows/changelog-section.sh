#!/bin/sh
# Print the CHANGELOG.md section for one version (Keep a Changelog
# layout: everything between "## [X.Y.Z]" and the next "## [" heading,
# minus the reference-link footer). Exits non-zero if the section is
# missing or empty: releasing a version the changelog does not
# describe is a mistake, not a formality.
#
# Usage: changelog-section.sh <version>
set -e

version="$1"
[ -n "$version" ] || { echo "usage: changelog-section.sh <version>" >&2; exit 2; }

notes=$(awk -v ver="$version" '
  # Version headings: "## X.Y.Z" or the bracketed "## [X.Y.Z]" style.
  /^## / { grab = ($2 == ver || index($0, "[" ver "]") > 0); next }
  /^\[.*\]: / { next }  # reference-link footer
  grab { print }
' CHANGELOG.md)

# Trim leading/trailing blank lines.
notes=$(printf '%s' "$notes" | awk 'NF {found=1} found' | awk '{lines[NR]=$0; if (NF) last=NR} END {for (i=1; i<=last; i++) print lines[i]}')

if [ -z "$notes" ]; then
  echo "CHANGELOG.md has no section for version $version — add one before releasing" >&2
  exit 1
fi
printf '%s\n' "$notes"
