from pytube import YouTube
import pytube.exceptions
import os
import csv
import traceback
import shutil
import source.config as config


def download(link, recursive_depth=0, max_depth=5):
    try:
        youtubeObject = YouTube(link)
        try:
            youtubeObject = youtubeObject.streams.get_highest_resolution()
        except pytube.exceptions.VideoUnavailable:
            print("Video is unavailable. Link: ", link)
            return None

        print("Downloading video...")
        youtubeObject.download()
        print("Video has been downloaded\n")
        return True
    except:
        traceback.print_exc()
        print("\nAn error has occurred with the link: ", link, "\n")
        if recursive_depth < max_depth:
            print("Trying again... Attempt: ", recursive_depth + 1, "\n")
            download(link, recursive_depth + 1)
        else:
            print("Max depth reached. Link: ", link)
            return None


def find_mp4_and_copy_to_folder(copy_file_name=None, move_to_folder=config.INPUT_FOLDER_PATH):
    for file in os.listdir():
        if file.endswith(".mp4"):
            file_name = file
            if copy_file_name is None:
                copy_file_name = file_name

            # make sure there is a .mp4 extension
            if not copy_file_name.endswith(".mp4"):
                copy_file_name += ".mp4"

            shutil.copyfile(
                (file),
                move_to_folder + copy_file_name
            )
            try:
                os.remove(file)
            except:
                return None
                pass
            print(f"File has been moved to the folder {move_to_folder.split(os.sep)[-2:]}. File name: ", file)
            return file_name

    return None


def get_links_from_csv(csv_file, link_column, move_to_folder):
    links = []
    list_rows = list(csv.reader(open(csv_file, 'r')))
    last_link = ""

    cnt = 0
    # find number of unique links
    for i, row in enumerate(list_rows):
        if i == 0 or len(row) < 2:
            continue
        try:
            if last_link != row[link_column]:
                last_link = row[link_column]
                links.append(last_link)
                cnt += 1
        except:
            traceback.print_exc()
            print("An error has occurred")

    for i, link in enumerate(links):
        try:
            print("link", link)
            print(f"Downloading video {i} of {len(links)}")
            download(link)
            copy_file_name = link.split("v=")[-1]
            find_mp4_and_copy_to_folder(copy_file_name=copy_file_name, move_to_folder=move_to_folder)
            cnt += 1
        except:
            traceback.print_exc()
            print("An error has occurred in link: ", link, " cnt: ", cnt)

    print("cnt", cnt)


if __name__ == '__main__':
    get_links_from_csv(config.INPUT_FOLDER_PATH + "mosei.csv", 1, move_to_folder=config.INPUT_FOLDER_PATH + "mosei" + os.sep)