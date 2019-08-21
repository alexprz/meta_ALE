import ntpath
import os
import pickle
from colorama import Fore, Style

def filename_from_path(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def pickle_dump(obj, file_path, save=True, verbose=True):
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        filename = filename_from_path(file_path)
        print(f'File {filename} {Fore.GREEN}dumped succesfully.{Style.RESET_ALL}')
    return obj


def pickle_load(file_path, verbose=True, load=True):
    if not load:
        print(f'{Fore.YELLOW}No file loaded.{Style.RESET_ALL}')
        return None

    filename = filename_from_path(file_path)
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print(f'File {filename} {Fore.GREEN}loaded succesfully.{Style.RESET_ALL}')
        return obj
    except OSError:
        print(f'File {filename} does not exist. {Fore.YELLOW}No file loaded.{Style.RESET_ALL}')
        return None
    except EOFError:
        print(f'File {filename} is empty. {Fore.YELLOW}No file loaded.{Style.RESET_ALL}')
        return None