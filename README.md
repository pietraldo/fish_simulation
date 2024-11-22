# Fish Simulation 🐟

## Ruch rybek

### 1. Unikanie krawędzi  
   Podczas zbliżania się do krawędzi dodawany jest `turnfactor` do prędkości boida.

### 2. Separation (Separacja)  
   Boidy oddalają się od siebie, aby unikać kolizji:
   ```
   for otherboid in neighbors_in_range:
       close_dx += boid.x - otherboid.x
       close_dy += boid.y - otherboid.y

   boid.vx += close_dx * avoidfactor
   boid.vy += close_dy * avoidfactor
   ```
### 3. Alignment  (Wyrównanie) 
   ```
  for otherboid in neighbors_in_range:
      xvel_avg += otherboid.vx
      yvel_avg += otherboid.vy
      neighboring_boids += 1

  xvel_avg /= neighboring_boids
  yvel_avg /= neighboring_boids
  
  boid.vx += (xvel_avg - boid.vx) * matchingfactor
  boid.vy += (yvel_avg - boid.vy) * matchingfactor
```

### 4. Cohesion (Spójność)
```
for otherboid in neighbors_in_range:
    xpos_avg += otherboid.x
    ypos_avg += otherboid.y
    neighboring_boids += 1

xpos_avg /= neighboring_boids
ypos_avg /= neighboring_boids

boid.vx += (xpos_avg - boid.x) * centeringfactor
boid.vy += (ypos_avg - boid.y) * centeringfactor
```


Algorytm
1. Przekopiowanie początkowych pozycji rybek do karty graficznej.
2. Uruchomienie symulacji, gdzie każdy wątek oblicza prędkość i kierunek swojej rybki na podstawie pozycji innych rybek w globalnej pamięci.
3. Zapisanie nowych pozycji na karcie graficznej.
4. Procesor kopiuje dane z karty graficznej i wyświetla rybki.





