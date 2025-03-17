import os
import argparse

def combine_markdown_files(input_dir, output_file):
    """
    Combines all markdown files in the input directory into a single markdown file.
    
    Args:
        input_dir (str): Path to the directory containing markdown files
        output_file (str): Path to the output file
    """
    # Get all markdown files in the directory
    markdown_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.md') or file.endswith('.markdown'):
            markdown_files.append(os.path.join(input_dir, file))
    
    # Sort files to maintain a consistent order
    markdown_files.sort()
    
    # Combine the content of all files
    combined_content = ""
    for file_path in markdown_files:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Add a header with the filename as separation
            combined_content += f"\n\n## File: {file_name}\n\n"
            combined_content += content
            combined_content += "\n\n---\n"
    
    # Write the combined content to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(combined_content)
    
    print(f"Combined {len(markdown_files)} markdown files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple markdown files into one")
    parser.add_argument("input_dir", help="Directory containing markdown files")
    parser.add_argument("output_file", help="Output markdown file")
    args = parser.parse_args()
    
    combine_markdown_files(args.input_dir, args.output_file)