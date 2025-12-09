#!/usr/bin/env python3
"""Fix generation bug in genome.py"""

# Read the file
with open('src/abe_nlp/genome.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "# Fitness tracking" and insert before it
new_lines = []
for i, line in enumerate(lines):
    if line.strip() == "# Fitness tracking" and i > 0 and lines[i-1].strip() == "":
        # Insert the sync line before the comment
        new_lines.append("        # Sync generation from metadata to base class\n")
        new_lines.append("        self.generation = self.metadata.generation\n")
        new_lines.append("\n")
    new_lines.append(line)

# Write back
with open('src/abe_nlp/genome.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Fixed generation sync bug")
