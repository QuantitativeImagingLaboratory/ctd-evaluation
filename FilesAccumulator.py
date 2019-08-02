import os

class FilesAccumulator:
    def __init__(self, dirname):
        self.dirname = dirname


    def find(self, extensions=['.avi', '.mp4', 'mov'], excludes=['variants']):

        list_of_finds = []
        for root, dirs, files in os.walk(self.dirname):

            for file in files:
                for ext in extensions:
                    if file.endswith(ext):
                        exclude_flag = False
                        full_file_name = os.path.join(root, file)
                        for exclude in excludes:
                            if full_file_name.find(exclude) != -1:
                                exclude_flag = True
                        if not exclude_flag:
                            list_of_finds.append(full_file_name)

        return list_of_finds