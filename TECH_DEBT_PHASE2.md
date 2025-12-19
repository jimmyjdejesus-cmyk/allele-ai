# Technical Debt Cleanup - Phase 2

## Medium Priority Issues to Address

### 🎯 **Target: Package Naming Consistency**

#### Files with Mixed "allele" vs "phylogenic" References:
- `README.md` - Likely has old allele references
- `docs/index.md` - Documentation may need updates
- Source code files - Check for remaining allele imports/names
- Comments and docstrings

### 🎯 **Target: Type Annotation Cleanup**

#### Current mypy Ignores to Address:
1. `phylogenic.observability.*` - 32 files in observability package
2. `phylogenic.llm_openai` - Missing imports
3. Tests excluded from type checking

### 🎯 **Target: Repository Structure Cleanup**

#### Additional Directories:
- `bin/` - Check contents and necessity
- `scripts/` - Review and organize
- `examples/` - Ensure current and relevant
- `launch_materials/` - Evaluate purpose

## Next Steps
1. **Audit mixed naming** across all files
2. **Update README.md** with consistent branding
3. **Review observability package** for type improvements
4. **Clean up additional directories**

## Estimated Effort
- **Mixed naming cleanup:** 2-3 hours
- **README updates:** 30 minutes
- **Type annotation start:** 4-6 hours
- **Directory cleanup:** 1-2 hours
