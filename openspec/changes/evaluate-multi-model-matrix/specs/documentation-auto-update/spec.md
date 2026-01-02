# Documentation Auto-Update

## ADDED Requirements

### REQ-DOC-001: Whitepaper Section Update
The system MUST automatically update Section 4.2.5 of the whitepaper with matrix evaluation results.

#### Scenario: Insert matrix results into whitepaper
Given matrix evaluation results
And whitepaper file docs/whitepaper/phylogenic_whitepaper.md
When update_whitepaper_benchmarks() is called
Then it locates Section 4.2.5 or creates it
And inserts results tables between markers
And preserves existing content outside markers
And validates Markdown syntax

#### Scenario: Handle missing markers
Given whitepaper without matrix results markers
When update is attempted
Then it creates Section 4.2.5 with markers
And inserts results content
And logs the addition

### REQ-DOC-002: README Matrix Table Update
The system MUST automatically update the README with a comprehensive matrix evaluation table.

#### Scenario: Update README matrix section
Given matrix evaluation results
And README.md file
When update_readme_matrix() is called
Then it locates matrix evaluation markers
And replaces content between markers with:
- Full Model × Personality × Benchmark matrix
- Summary statistics
- Best performers
And preserves existing content outside markers

#### Scenario: Handle missing README markers
Given README without matrix evaluation markers
When update is attempted
Then it creates a new section with markers
And inserts matrix table
And logs the addition

### REQ-DOC-003: Marker-Based Updates
The system MUST use marker comments to identify update locations in documentation files.

#### Scenario: Whitepaper markers
Given whitepaper file
When markers are defined
Then they use format:
- `<!-- MATRIX_RESULTS_START -->`
- `<!-- MATRIX_RESULTS_END -->`
And markers are placed in Section 4.2.5

#### Scenario: README markers
Given README file
When markers are defined
Then they use format:
- `<!-- MATRIX_EVALUATION_START -->`
- `<!-- MATRIX_EVALUATION_END -->`
And markers are placed in appropriate section

### REQ-DOC-004: Content Preservation
The system MUST preserve all content outside marker boundaries when updating documentation.

#### Scenario: Preserve existing content
Given whitepaper with content before and after markers
When update is performed
Then content before `<!-- MATRIX_RESULTS_START -->` is preserved
And content after `<!-- MATRIX_RESULTS_END -->` is preserved
And only content between markers is replaced

#### Scenario: Handle missing end marker
Given file with start marker but no end marker
When update is attempted
Then it logs an error
And does not modify the file
And suggests adding end marker

### REQ-DOC-005: Markdown Validation
The system MUST validate Markdown syntax after updates to ensure documentation remains valid.

#### Scenario: Validate updated Markdown
Given updated whitepaper or README
When validation is performed
Then it checks:
- Table syntax is correct
- Headings are properly formatted
- Links are valid
- No broken Markdown structures
And reports any issues

#### Scenario: Handle validation failures
Given Markdown with syntax errors after update
When validation fails
Then it logs errors
And optionally reverts changes
And reports issues to user

### REQ-DOC-006: Backup Before Update
The system MUST create backups of documentation files before updating them.

#### Scenario: Create backup before update
Given whitepaper file to update
When update is initiated
Then it creates a backup file (e.g., .bak or timestamped)
And stores original content
And proceeds with update

#### Scenario: Restore from backup
Given a failed update
And backup file exists
When restore is requested
Then it restores original content from backup
And removes backup file
And logs the restoration

### REQ-DOC-007: Idempotent Updates
The system MUST support idempotent updates (safe to run multiple times with same results).

#### Scenario: Run update multiple times
Given same matrix results
When update is run multiple times
Then each run produces identical output
And markers remain in same location
And content is consistently formatted

#### Scenario: Handle concurrent updates
Given multiple update processes
When updates run concurrently
Then file locking or atomic writes prevent corruption
And last write wins or merge conflict is handled
And backups are created for each attempt

## MODIFIED Requirements

### REQ-DOC-008: Extend Existing Update Scripts
The existing update scripts (e.g., update_readme_benchmarks.py) MUST be extended or new scripts MUST be created following similar patterns.

#### Scenario: Follow existing update patterns
Given existing update_readme_benchmarks.py
When creating matrix update scripts
Then they follow similar structure:
- Marker-based updates
- Backup creation
- Validation
- Error handling
And maintain consistency

