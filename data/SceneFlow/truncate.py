import sys

def remove_lines_with_string(file_path, string_to_remove):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if string_to_remove not in line:
                print(line, end='')
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python truncate.py <file_path> <string_to_remove>")
    else:
        file_path = sys.argv[1]
        string_to_remove = sys.argv[2]

        remove_lines_with_string(file_path, string_to_remove)