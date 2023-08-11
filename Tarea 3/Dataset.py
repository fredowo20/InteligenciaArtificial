import pandas as pd

anime = pd.read_csv('Tarea 3/anime.csv')
#print(anime)

ds = anime[["Genres", "Score"]]
#print(ds)

ds2 = ds[(ds["Score"].notna()) & (ds["Genres"] != "Unknown")] # se elimina data innecesaria
#print(ds2)

arrGenre = []
arrScore = []               # se definen arreglos
generos = []

for columna in ds2["Genres"]:  # se recorre los generos
    columna = columna.split(',')  # se separa por comas
    for ndx, col in enumerate(columna):
        columna[ndx] = str(col).lstrip(' ')
    arrGenre.append(columna) # se agregan los generos
    for gen in columna:
        gen = gen.strip() # se eliminan los espacios
        if gen not in generos:
            generos.append(gen) # se agregan nuevos generos unicos
        
for columna in ds2["Score"]:
    arrScore.append(columna)

tmp2 = []

for arr in arrGenre:
    tmp1 = []
    for gnr in generos:
        if gnr in arr:
            tmp1.append(1)
        else:
            tmp1.append(0)
    tmp2.append(tmp1)

# for count in range(len(arrGenre)):
#     print(generos)
#     print(tmp2[count])
#     print(arrGenre[count])
#     print(" ")

df = pd.DataFrame(tmp2, columns=['Action', 'Drama', 'Supernatural', 'Suspense', 'Adventure', 'Fantasy', 'Comedy', 'Romance', 'Horror', 'Sci-Fi', 'Ecchi', 'Mystery', 'Sports', 'Award Winning', 'Avant Garde', 'Slice of Life', 'Gourmet', 'Boys Love', 'Girls Love', 'Hentai', 'Erotica'])
df['Score'] = arrScore
df.to_csv('animeFinal.csv')
print(df)