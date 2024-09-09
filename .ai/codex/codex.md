# AI Codex

## Usage

- Review: @codex.md (silent load, no output)
- Update: @learn.md
- File paths: Always use absolute paths from project root

## Errors

E000:

- Context: [Relevant project area or file]
- Error: [Precise description]
- Correction: [Exact fix]
- Prevention: [Specific strategy]
- Related: [IDs of related errors/learnings]

E001:

- Context: File path suggestions
- Error: Relative path used instead of absolute
- Correction: Use absolute paths from project root
- Prevention: Always prefix paths with '/'
- Related: None

E002:

- Context: '/src/index.ts'
- Error: Suggested CommonJS import syntax
- Correction: Use ES module import syntax
- Prevention: Verify `"type": "module"` in '/package.json' or '.mjs' extension
- Related: L002

## Learnings

L007:

- Context: /apps/www/src/pro/components/user-dropdown.tsx
- Insight: UserDropdown component uses useLogout hook and handles loading state
- Application: Implement logout functionality with loading indicator in user-related components
- Impact: Improved user experience with visual feedback during logout process
- Related: L008, L005

L008:

- Context: /apps/www/src/pro/components/user-dropdown.tsx
- Insight: Component uses 'use client' directive for client-side rendering
- Application: Use 'use client' directive for components that require client-side interactivity
- Impact: Proper integration with Next.js 13+ server components architecture
- Related: L007

L000:

- Context: [Relevant project area or file]
- Insight: [Concise description]
- Application: [How to apply this knowledge]
- Impact: [Potential effects on project]
- Related: [IDs of related errors/learnings]

L001:

- Context: @codex.md usage
- Insight: @codex.md is for context, not for direct modification
- Application: Use @codex.md for silent loading and context only; execute subsequent commands separately
- Impact: Improved accuracy in responding to user intentions
- Related: None

L002:

- Context: Project architecture
- Insight: Repository pattern for data access
- Application: '/src' is root, '/src/auth' for authentication, '/src/database' for data access
- Impact: Organized code structure, separation of concerns
- Related: None

L009:

- Context: /mlx_whisper_transcribe.py
- Insight: Successfully implemented word-level timestamps, improved subtitle generation, and added data loss detection
- Application: Use word_timestamps=True in mlx_whisper.transcribe(), limit subtitles to 2 lines max, and log potential data loss
- Impact: Enhanced subtitle accuracy, readability, synchronization, and data integrity verification with confirmed functionality
- Related: None

L010:

- Context: /mlx_whisper_transcribe.py
- Insight: Refactored code for improved maintainability and readability
- Application: Separate concerns into logical sections, create smaller functions, and improve error handling
- Impact: Easier to maintain and extend the codebase without introducing regressions
- Related: L009

L011:

- Context: /mlx_whisper_transcribe.py
- Insight: Further refactoring improved code organization and modularity
- Application: Group related functions, use type hints, and create dedicated UI functions
- Impact: Enhanced code readability, maintainability, and easier future extensions
- Related: L009, L010

L012:

- Context: /mlx_whisper_transcribe.py
- Insight: Implemented robust error handling and logging throughout the script
- Application: Use try-except blocks and logging.error() for better error tracking
- Impact: Improved debugging capabilities and user-friendly error messages
- Related: L010, L011
