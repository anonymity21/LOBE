
def get_commit_list(commit_file='//MLIR/llvm-release/llvm-project/mlir.log'):
    # Read the commit log file
    with open(commit_file, 'r') as file:
        content = file.readlines()
    commit_list = []
    for line in content:
        if line.startswith('commit'):
            commit_list.append(line.split(' ')[1].strip())
    return commit_list


def main(start_commit, end_commit, commit_list):
    # Ensure the commit list is ordered if it's not already
    # commit_list.sort()  # Uncomment this if you need sorting based on some criteria

    # Find the positions of start_commit and end_commit manually
    start_index = -1
    end_index = -1
    
    # Iterate through the commit_list to find the indices of start_commit and end_commit
    for i, commit in enumerate(commit_list):
        if commit == start_commit:
            start_index = i
        if commit == end_commit:
            end_index = i
    
    # Check if both start_commit and end_commit were found in the list
    if start_index == -1 or end_index == -1:
        print("Start commit or end commit not found in the commit list.")
        return None

    # Ensure that start_commit comes before end_commit
    if start_index > end_index:
        print("Start commit should be before end commit.")
        return None

    # Find the middle commit by calculating the average index
    middle_index = (start_index + end_index) // 2

    print(f"Start commit index: {start_index}")
    print(f"End commit index: {end_index}")     
    print(f"Middle commit index: {middle_index}")

    # Return the middle commit
    middle_commit = commit_list[middle_index]
    print(f"The middle commit between {start_commit} and {end_commit} is {middle_commit}")
    return middle_commit


start_commit = "ea488bd6e1f7bf52d6ec7a40c7116670f06e92a6"
end_commit = "2f9f9afa4e1281b4ac7c8ad36860a4e35e6f5070"
main(start_commit, end_commit, get_commit_list())