import os
import time
import requests

from typing import *

import os.path


def download_episode_by_id(_id : int, dir : str):
    path = os.path.join(dir, str(_id) + '.json')
    if os.path.isfile(path):
        print('Skip id {} since file already exists'.format(_id))
        return
    j = requests.get('https://www.kaggleusercontent.com/episodes/{}.json'.format(_id))
    if j.status_code != 200:
        print("Couldn't download submission with id {}".format(_id))
        return
    with open(path, "w") as f:
        f.write(j.content.decode())

base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
list_url = base_url + "ListEpisodes"


def get_episode_replay(episode_id: int):
    body = {
        "EpisodeId": episode_id
    }

    response = requests.post(get_url, json=body)
    return response.json()


def list_episodes(episode_ids: List[int]):
    return __list_episodes({
        "Ids": episode_ids
    })


def list_episodes_for_team(team_id: int):
    return __list_episodes({
        "TeamId": team_id
    })


def list_episodes_for_submission(submission_id: int):
    return __list_episodes({
        "SubmissionId": submission_id
    })


def __list_episodes(body):
    response = requests.post(list_url, json=body)
    return response.json()


def download_all_sessions_for_submission(submission_id : int, dir : str, delta : int = 0.05):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)
    episodes = list_episodes_for_submission(submission_id)
    if not episodes['wasSuccessful']:
        print('Not successfull download')
        return
    for e in episodes['result']['episodes']:
        _id = e['id']
        download_episode_by_id(_id, dir)
        time.sleep(delta)
