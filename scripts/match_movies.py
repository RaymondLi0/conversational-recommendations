from tqdm import tqdm
import csv
import re
import argparse
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def merge_indexes(matched_db_path, movielens_path, write_to):
    """
    merge both movie files into a single file: globalId, movieName, dbId, movielensId
    movies from the DB are put first. Then are added the remaining movies from movielens
    :param matched_db_path:
    :param movielens_path:
    :param write_to:
    :return:
    """
    movielens = read_csv(movielens_path)
    matched_db = read_csv(matched_db_path)

    merged = [[movie[0], db_id, movie[1]] for db_id, movie in matched_db.items()]

    # Remember all movies from movielens that have no match in our list of mentioned movies
    to_add = {movielensId: True for movielensId in movielens}
    for db_id, movie in matched_db.items():
        if int(movie[1]) != -1:
            to_add[int(movie[1])] = False

    for movielensId in movielens:
        if to_add[movielensId]:
            # there is no db_id for those movies
            merged.append([movielens[movielensId][0], -1, movielensId])

    with open(write_to, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'movieName', 'databaseId', 'movielensId'])
        for i, movie in enumerate(merged):
            writer.writerow([i] + movie)


def read_csv(path):
    """

    :param path:
    :return: (int) movieId  -> (array) rest of the row
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        id2movie = {int(row[0]): row[1:] for row in reader if row[0] != 'movieId'}
    return id2movie


def get_movies_db(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        id2movie = {}
        for row in reader:
            if row[0] != 'movieId':
                # separate the title into movieName and movieYear if present
                pattern = re.compile('(.+)\((\d+)\)')
                match = re.search(pattern, row[1])
                if match is not None:
                    # movie Year found
                    content = (match.group(1).strip(), match.group(2))
                else:
                    # movie Year not found
                    content = (row[1].strip(), None)
                id2movie[int(row[0])] = content
    print("loaded {} movies from {}".format(len(id2movie), path))
    return id2movie


def make_name(name, year):
    if year is None:
        return name
    if int(year) >= 1900:
        return name + " (" + year + ")"
    return name


def find_in_file(db_path, movielens_path, write_to="movies_matched.csv"):
    """
    For each movie in db, find equivalent movie name in movielens.
    Writes to a csv file: dbId, movieName, movielensId
    :param db_path:
    :param movielens_path:
    :param write_to:
    :param manual_match:
    :return:
    """
    movielens = get_movies_db(movielens_path)
    movies_db = get_movies_db(db_path)
    matched_movies = {}

    total_exact_matches = 0
    total_movie_not_matched = 0
    for movieId, (db_name, year) in tqdm(movies_db.items()):
        # Pre-process name
        processed_name = db_name.strip()
        processed_name = processed_name.replace("&", "and")
        # Remove "The" at the beginning to avoid format problems (like "Avengers, The (2012)")
        if processed_name.startswith("The "):
            processed_name = processed_name[4:]
        if processed_name.startswith("A "):
            processed_name = processed_name[2:]
        found = 0

        for i, (movielensId, (movielens_name, movielens_year)) in enumerate(movielens.items()):
            movielens_name = movielens_name.replace("&", "and")
            # search for exact same name, or appended with ", The". Year has to be None, or the same
            if (processed_name == movielens_name
                or processed_name + ", The" == movielens_name
                or "The " + processed_name == movielens_name
                or processed_name + ", A" == movielens_name
                or "A " + processed_name == movielens_name) \
                    and (movielens_year is None or year is None or movielens_year == year):
                found = 1
                matched_movies[movieId] = (db_name, year, movielensId, (movielens_name, movielens_year))
                total_exact_matches += 1
                break
        if found == 0:
            total_movie_not_matched += 1
    print("Over {} movies mentioned in ReDial, {} of them are perfectly matched, {} of them have no match in movielens"
        .format(len(movies_db), total_exact_matches, total_movie_not_matched))

    with open(write_to, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['movieId', 'movieName', 'movielensId'])
        for key, val in movies_db.items():
            movielensId = matched_movies[key][2] if key in matched_movies else -1
            writer.writerow([key, make_name(*val), movielensId])

    return matched_movies


if __name__ == '__main__':
    # intermediate file used to match the movie names.
    intermediate_file = "movies_matched.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--redial_movies_path")
    parser.add_argument("--ml_movies_path")
    parser.add_argument("--destination", default="data/movies_merged.csv")
    args = parser.parse_args()
    _ = find_in_file(args.redial_movies_path,
                     args.ml_movies_path,
                     write_to=intermediate_file)
    merge_indexes(intermediate_file, args.ml_movies_path, args.destination)
    os.remove(intermediate_file)
