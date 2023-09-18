'''
create a dataset with audio features of tracks from a given playlist
inspired by https://github.com/Brice-Vergnou/spotify_recommendation
'''

import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace')
np.set_printoptions(precision=3, suppress=True)


def get_playlist_features(token, playlist_id):
    '''retrieve audio features for the tracks from the given playlist'''
    headers = {'Accept': 'application/json',
               'Content-Type': 'application/json',
               'Authorization': f'Bearer {token}'}

    # obtain tracks from the given playlist
    playlist_url = f'https://api.spotify.com/v1/playlists/{playlist_id}'
    name_flag = '?fields=name'
    tracks_flag = '/tracks?fields=items(track(id%2Cname%2Cartists(name)))'
    tracks_res = requests.get(playlist_url + tracks_flag, headers=headers)
    try:
        print(tracks_res.json()['error']['message'])
        print('generate new token at https://developer.spotify.com/console/get-current-user/')
        return None, None
    except:
        playlist_name = requests.get(playlist_url + name_flag, headers=headers).json()['name']
    tracks = []
    for t in tracks_res.json()['items']:
        try:
            tracks.append({'id': t['track']['id'],
                           'name': t['track']['name'],
                           'artist': t['track']['artists'][0]['name']})
        except:
            pass
    tracks = pd.DataFrame(tracks)

    # obtain audio features
    features_url = f'https://api.spotify.com/v1/audio-features?ids={"%2C".join(tracks["id"])}'
    features_res = requests.get(features_url, headers=headers)
    features = []
    for t in features_res.json()['audio_features']:
        try:
            features.append({'id': t['id'],
                             'acousticness': t['acousticness'],
                             'danceability': t['danceability'],
                             'energy': t['energy'],
                             'instrumentalness': t['instrumentalness'],
                             'liveness': t['liveness'],
                             'loudness': t['loudness'] / 60 + 1,
                             'mode': float(t['mode']),
                             'speechiness': t['speechiness'],
                             'tempo': t['tempo'] / 200,
                             'valence': t['valence']})
        except:
            pass
    features = pd.DataFrame(features)

    return tracks, features, playlist_name


def download_spotify_data(playlists, save_name='spotify_data'):
    '''download and save audio features for the tracks from the provided playlists'''
    os.makedirs('./data/', exist_ok=True)
    info = []
    for name, id in playlists.items():
        downloaded = False
        while not downloaded:
            try:
                tracks, features, playlist_name = get_playlist_features(token, id)
                features.to_csv(f'./data/{name}.csv')
                info.append({'genre': name, 'spotify_id': id, 'spotify_name': playlist_name, 'tracks': len(tracks)})
                print(f'downloaded data from {name}')
                downloaded = True
            except:
                print(f'could not download data from {name} -- trying again in 30 sec...')
                time.sleep(30)
    info = pd.DataFrame(info)
    info.to_csv(f'./{save_name}.csv')
    return info


def read_spotify_data(info):
    '''compute average feature vector for each data file in info'''
    genres, tracks = {}, []
    for genre in info['genre']:
        tracks.append(pd.read_csv(f'./data/{genre}.csv', index_col=0))
        genres[genre] = dict(tracks[-1].mean())
    tracks = pd.concat(tracks, ignore_index=True)
    genres = pd.DataFrame(genres).transpose().sort_index()
    return genres, tracks


def plot_genre_features(data):
    '''plot average feature vector for each genre'''
    features = np.array(data.values.tolist()).T
    fig, ax = plt.subplots(figsize=(8,5))
    plt.imshow(features)
    ax.set_xticks(range(data.shape[0]))
    ax.set_xticklabels(data.index, rotation=45, ha='left', rotation_mode='anchor')
    ax.xaxis.tick_top()
    ax.set_yticks(range(data.shape[1]))
    ax.set_yticklabels(data.columns)
    plt.grid(None)
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()


def plot_genre_diffs(genres):
    '''plot norm of differences of average feature of genres'''
    data = np.array(genres.values.tolist())
    feature_similarity = np.array([[np.linalg.norm(data[i] - data[j])
                                    for i in range(len(data))] for j in range(len(data))])
    fig, ax = plt.subplots(figsize=(8,6))
    plt.imshow(feature_similarity)
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres.index, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres.index)
    plt.grid(None)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # generate new token at https://developer.spotify.com/documentation/web-api/concepts/access-token
    token = 'BQB6_ZDp3nwtTAbaAA3csiN6dJqFfArrT_T9QJ'\
          + 'nT2HM1H9ITiLyGQaE4avcEJeoHidT4yZWiPCgD'\
          + 'Yc1lhvVHrJyM00d2UTMZEvKIs5YPC4iNvnbtw4w'

    playlists_test = {'top50': '37i9dQZEVXbMDoHDwVN2tF'}
    playlists_train = {'acoustic': '37i9dQZF1DX504r1DvyvxG',
                       'alternative': '37i9dQZF1DX82GYcclJ3Ug',
                       'ambient': '37i9dQZF1DWUrPBdYfoJvz',
                       'blues': '37i9dQZF1DXd9rSDyQguIk',
                       'chill': '37i9dQZF1DX0MLFaUdXnjA',
                       'classical': '37i9dQZF1DWWEJlAGA9gs0',
                       'country': '37i9dQZF1DWVpjAJGB70vU',
                       'electro': '37i9dQZF1DX4dyzvuaRJ0n',
                       'folk': '37i9dQZF1DWVmps5U8gHNv',
                       'gospel': '37i9dQZF1DXcb6CQIjdqKy',
                       'hiphop': '37i9dQZF1DX2RxBh64BHjQ',
                       'instrumental': '37i9dQZF1DX4sWSpwq3LiO',
                       'jazz': '37i9dQZF1DX7YCknf2jT6s',
                       'kpop': '37i9dQZF1DX14fiWYoe7Oh',
                       'metal': '37i9dQZF1DX9qNs32fujYe',
                       'pop': '37i9dQZF1DX4JAvHpjipBk',
                       'punk': '37i9dQZF1DXasneILDRM7B',
                       'rock': '37i9dQZF1DWXRqgorJj26U',
                       'rnb': '37i9dQZF1DWUFAJPVM3HTX',
                       'soul': '37i9dQZF1DX9XIFQuFvzM4'}

    # download or load spotify data
    download = False
    if download:
        info_test = download_spotify_data(playlists_test, save_name='spotify_actions')
        info_train = download_spotify_data(playlists_train, save_name='spotify_genres')
    else:
        info_test = pd.read_csv('./spotify_actions.csv', index_col=0)
        info_train = pd.read_csv('./spotify_genres.csv', index_col=0)
    genres, tracks_train = read_spotify_data(info_train)
    _, tracks = read_spotify_data(info_test)

    # visualize data
    plot_genre_features(genres)
    plot_genre_diffs(genres)

