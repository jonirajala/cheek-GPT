import os
from lyricsgenius import Genius
from random import shuffle

def combine_lyrics():
    data_dir = "lyrics"
    output_dir = "raw_rapdataset.txt"
    with open(output_dir , 'w') as output_f:
        dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        shuffle(dirs)
        for dir in dirs:
            print(dir)
            artist_dir = os.path.join(data_dir, dir, "placeholder_album", )
            files = [f for f in os.listdir(artist_dir) if os.path.isfile(os.path.join(artist_dir, f))]
            for file in files:                    
                if file.endswith(".txt"):
                    with open(os.path.join(artist_dir, file), "r") as input_f:
                        output_f.write(input_f.read().replace("<br />", " "))

def scrape_lyrics():
    token = "hn-fXDODUZ1p4Laful9xzmYV6GdDPkReGtmRyedcoCyGQtgmlp9Q0RvZmqCu8gAB"

    genius = Genius(token)
    # artists = ["william (FIN)", "Sexmane", "Cledos", "ibe", "Niko Katavainen", "Turisti", "Fabe (FIN)", "Costi (FIN)", "Lauri Haav", "Ghettomasa", "Cheek (FIN)",  "DeezyDavid", "Shrty", "Blacflaco", "Nebi", "Ege Zulu", "Elastinen", "Gracias", "Karri Koira", "Ruudolf", "Mäk Gälis", "Paperi-T", "TheoFuego", "Jore & Zpoppa", "Axel Kala", ""]
    artists = ["Niko Katavainen", "Karri Koira", "Ruudolf", "Mäk Gälis", "Paperi-T", "TheoFuego", "Jore & Zpoppa", "Axel Kala"]
    # test Niko Katavainen
    for artist_name in artists:
        artist = genius.search_artist(artist_name, sort="title", allow_name_change=False)
        if not artist:
            print(f"Artist {artist} not found")
            continue
        album_path = os.path.join("lyrics", artist.name.replace("\u200b", ""), "placeholder_album")
        if not os.path.exists(album_path):
            os.mkdir(os.path.join("lyrics", artist.name.replace("\u200b", "")))
            os.mkdir(album_path)
        
        songs = artist.songs
        for song in songs:
            try:
                title = song.title.replace("/", " ")
                fn = f"{title}.txt"
                song_path = os.path.join(album_path, fn)
                if not os.path.exists(song_path):
                    print(song)
                    try:
                        lyrics = song.to_text().split("\n", 1)[1]
                    except IndexError:
                        continue
                    waste_lines = ["[Chorus]\n", "[Bridge]\n", "[Hook]\n", "[Verse 1]\n", "[Verse 2]\n", "[Kertosäe]\n", "[Outro]\n", "[Verse]\n"]
                    for line in waste_lines:
                        lyrics = lyrics.replace(line, "")
                    with open(song_path, 'w') as file:
                        file.write(lyrics)
            except Exception as e: 
                print(f"Error occured in song {song}")
                print(e)


if __name__ == "__main__":
    scrape_lyrics()
    # combine_lyrics()