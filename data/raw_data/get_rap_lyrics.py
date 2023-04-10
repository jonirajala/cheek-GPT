import os
from lyricsgenius import Genius
from random import shuffle

def combine_lyrics():
    data_dir = "lyrics"
    output = []
    dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for dir in dirs:
        print(dir)
        artist_dir = os.path.join(data_dir, dir, "placeholder_album", )
        files = [f for f in os.listdir(artist_dir) if os.path.isfile(os.path.join(artist_dir, f))]
        for file in files:                    
            if file.endswith(".txt"):
                with open(os.path.join(artist_dir, file), "r") as input_f:
                    output.append(input_f.read().replace("Embed", "").replace("You might also like", ""))
    
    shuffle(output)
    output_dir = "raw_rapdataset.txt"
    with open(output_dir , 'w') as output_f:
        output_f.write("\n".join(output)) # You can remove the replaces if you run the scrape_lyrics from zero


def scrape_lyrics():
    token = "hn-fXDODUZ1p4Laful9xzmYV6GdDPkReGtmRyedcoCyGQtgmlp9Q0RvZmqCu8gAB"
    genius = Genius(token)
    
    artists = ["william (FIN)", "Sexmane", "Cledos", "ibe", "Niko Katavainen", "Turisti", "Fabe (FIN)", "Costi (FIN)", "Lauri Haav", "Ghettomasa", "Cheek (FIN)",  "DeezyDavid", "Shrty", "Blacflaco", "Nebi", "Ege Zulu", "Elastinen", "Gracias", "Karri Koira", "Ruudolf", "Mäk Gälis", "Paperi-T", "TheoFuego", "Jore & Zpoppa", "Axel Kala", "Nuteh Jonez", "Yutu Brown", "Kerim Muslah", "SKII6", "Ade Iduozee", " Jami Kiskonen", "Madboiali", "Jami Faltin", "Ares (FIN)", "LILBRO (FI)", "Sebastian Noto", "Nino Mofu", "Jimi Vilppola", "Aaro630"]
    for artist_name in artists:
        artist = genius.search_artist(artist_name, sort="title", allow_name_change=False)
        if not artist:
            print(f"Artist {artist} not found")
            continue
        album_path = os.path.join("lyrics", artist.name.replace("\u200b", ""), "placeholder_album")
        if not os.path.exists(album_path):
            os.mkdir(os.path.join("lyrics", artist.name.replace("\u200b", "")))
            os.mkdir(album_path)
        
        for song in artist.songs:
            try:
                fn = f"{song.title.replace('/', ' ')}.txt"
                song_path = os.path.join(album_path, fn)
                if not os.path.exists(song_path):
                    lyrics = song.to_text().split("\n", 1)[1]
                    waste_lines = ["[Chorus]\n", "[Bridge]\n", "[Hook]\n", "[Verse 1]\n", "[Verse 2]\n", "[Kertosäe]\n", "[Outro]\n", "[Verse]\n", "Embed"]
                    for line in waste_lines:
                        lyrics = lyrics.replace(line, "")
                    with open(song_path, 'w') as file:
                        file.write(lyrics)
            except Exception as e: 
                print(f"Error occured in song {song}")
                print(e)


if __name__ == "__main__":
    scrape_lyrics()
    combine_lyrics()