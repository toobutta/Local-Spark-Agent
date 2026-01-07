#!/usr/bin/env python3
"""
Fix TypeScript errors in home.tsx by consolidating the handleCommand function
and removing orphaned code blocks.
"""

def fix_home_tsx():
    input_file = r"C:\Users\Andrew\DEVELLO\SparkPlug\SparkPlug\client\src\pages\home.tsx"
    output_file = r"C:\Users\Andrew\DEVELLO\SparkPlug\SparkPlug\client\src\pages\home.tsx"

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the line numbers for key sections
    handle_command_start = None
    handle_command_end = None
    unreachable_return_start = None
    orphaned_code_start = None
    orphaned_code_end = None
    duplicate_loading_start = None
    proper_return_start = None

    for i, line in enumerate(lines):
        # Find the first handleCommand function
        if 'const handleCommand = async (cmd: string) =>' in line and handle_command_start is None:
            handle_command_start = i

        # Find the first closing brace of handleCommand (the incomplete one)
        if handle_command_start is not None and handle_command_end is None:
            if i > handle_command_start and line.strip() == '};' and 'handleCommand' in ''.join(lines[handle_command_start:i]):
                handle_command_end = i

        # Find the unreachable MatrixLoader return
        if 'return (' in line and handle_command_end is not None and i > handle_command_end and unreachable_return_start is None:
            if 'MatrixLoader' in lines[i+1] if i+1 < len(lines) else False:
                unreachable_return_start = i

        # Find the orphaned command code (starts with command === "browser")
        if 'else if (command === "browser"' in line and orphaned_code_start is None:
            orphaned_code_start = i - 1  # Include the closing )); from previous block

        # Find where orphaned code ends (the duplicate loading check)
        if orphaned_code_start is not None and orphaned_code_end is None:
            if 'if (loading)' in line and i > orphaned_code_start:
                orphaned_code_end = i - 2  # Exclude the empty lines before

        # Find the proper return statement
        if 'return (' in line and i > 500:
            proper_return_start = i
            break

    print(f"handleCommand start: {handle_command_start}")
    print(f"handleCommand end: {handle_command_end}")
    print(f"unreachable return start: {unreachable_return_start}")
    print(f"orphaned code start: {orphaned_code_start}")
    print(f"orphaned code end: {orphaned_code_end}")
    print(f"proper return start: {proper_return_start}")

    # Build the fixed file
    fixed_lines = []

    # Add everything up to and including the handleCommand start
    fixed_lines.extend(lines[:handle_command_start + 3])  # Through "try {"

    # Add the orphaned command handling code
    if orphaned_code_start and orphaned_code_end:
        # Get the orphaned code
        orphaned_code = lines[orphaned_code_start:orphaned_code_end]

        # Find where the actual if/else chain starts (after the first closing ))
        chain_start = 0
        for i, line in enumerate(orphaned_code):
            if 'else if (command === "browser"' in line:
                chain_start = i
                break

        # Add the command handling chain
        fixed_lines.extend(orphaned_code[chain_start:])

    # Add the error handling and closing
    fixed_lines.append('    } catch (err) {\n')
    fixed_lines.append('      addLog("error", "SYSTEM ERROR: COMMAND EXECUTION FAILED");\n')
    fixed_lines.append('    } finally {\n')
    fixed_lines.append('      setIsProcessing(false);\n')
    fixed_lines.append('    }\n')
    fixed_lines.append('  };\n')
    fixed_lines.append('\n')

    # Add the loading check and proper return
    if proper_return_start:
        # Find the loading check (should be a few lines before proper_return_start)
        loading_check_start = proper_return_start - 7
        fixed_lines.extend(lines[loading_check_start:proper_return_start])
        fixed_lines.extend(lines[proper_return_start:])

    # Write the fixed content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

    print(f"\nFixed! Wrote {len(fixed_lines)} lines to {output_file}")

if __name__ == '__main__':
    fix_home_tsx()
