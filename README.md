my-psg-challenge
==============================

## Problématique

On possède une suite de 10 minutes de séquences (_n_ séquences) consécutives de foot, tirées au hasard, on souhaite:

1. Prédire la position _X_ et _Y_ des évènements n+1 à n+5
2. Prédire la possession du ballon à n+1
3. Prédire le type d'évènement à n+1


*Exemple d'une séquence à prédire:*
```r
     game_id event_period_id event_type_id event_team_id event_x event_y event_order
 1:  853139               1             1           148    50.2    49.3           1
 2:  853139               1             1           148    35.3    59.4           2
 3:  853139               1             5           143    24.4   101.5           3
 4:  853139               1             1           143    21.9   100.0           4
 5:  853139               1            61           143    22.7    87.4           5
 6:  853139               1             1           143    22.1    78.8           6
 7:  853139               1             1           143    17.1    83.0           7
 8:  853139               1             1           148    20.5    11.0           8
 9:  853139               1             1           148     4.9    27.0           9
10:  853139               1             3           148    51.9    92.3          10
```

*Exemple de la prédiction:*
```r
     game_id event_period_id event_type_id event_team_id event_x event_y event_order
11:  853139               1             1           143    33.0    18.9          11
12:  853139               1             1           143    42.3    21.3          12
13:  853139               1             1           143    27.5    46.6          13
14:  853139               1            61           143    28.8    39.1          14
15:  853139               1             5           148    68.5   101.6          15
```

## Réponse en trois modèles

1. Modèle de position de l'évènement pour n+1
2. Modèle de position des évènements n+2 à n+5
3. Modèle de possession du ballon

Le type d'évènement étant majoritairement des passes, aucun modèle ne sera testé.

Bien pensé à implémenté les règles du foot dans la prédiction (replacement du ballon, sortie de balle, enchainement fatal de certaines actions...)

## Scripts

L'essentiel du code est lancé par le `Makefile`.
## Le challenge de base

Opta : https://www.agorize.com/fr/challenges/xpsg?lang=fr

## Les données à disposition

* Données joueurs
* Données de matchs : un xml à la maille qualifier par match

## Données transformées pour le challenge

* Données de matchs d'apprentissage en csv, à la maille event: `games_train_events.csv` :
  * `jointure_key`: champ original 
  * `game_id`: identifiant du match
  * `game_date`: date du match
  * `away_team_id`: identifiant de l'équipe visiteur 
  * `away_team_name`: nom de l'équipe visiteur
  * `home_team_id`: identifiant de l'équipe à domicile
  * `home_team_name`: nom de l'équipe à domicile
  * `matchday`: journée ligue 1
  * `period_1_start`: horaire du début de période 1
  * `period_2_start`: horaire du début de la période 2
  * `event_id`: identifiant de l'événement
  * `event_type_id`: type d'événement
  * `event_assist`: Will only appear on an event if this event led directly to a goal
  * `event_keypass`: Will only appear on an event if this event led directly to a shot off target, blocked orsaved
  * `event_last_modifier`: horaire modification de l'événement
  * `event_min`: minute de l'événement (par rapport au coup d'envoi)
  * `event_outcome`: résultat de l'événement
  * `event_period_id`: periode (mi temps 1 ou 2)
  * `event_player_id`: identifiant du joueur
  * `event_sec`: seconde de l'événement (par rapport au coup d'envoi)
  * `event_team_id`: identifiant de l'équipe
  * `event_timestamp`: horaire de l'événement
  * `event_version`: identifiant de version de l'événement
  * `event_x`: position en abscisse de l'événement 
  * `event_y`: position en ordonnée de l'événement
  * `event_seconds_elapsed`: nombre de seconde écoulé depuis le coup d'envoi
  * `zone_id`: identifiant de la zone
  * `zone_name`: nom de la zone détaillée
  * `big_zone_name`: nom de la zone large
  * `event_order`: ordre des événements entre eux
  
* Données de matchs de test en csv, à la maille event: `games_test_events.csv` :
  * `game_id`: identifiant du match
  * `away_team_id`: identifiant de l'équipe visiteur 
  * `home_team_id`: identifiant de l'équipe à domicile
  * `matchday`: journée ligue 1
  * `event_period_id`: periode (mi temps 1 ou 2)
  * `event_type_id`: type d'événement
  * `event_min`: minute de l'événement (par rapport au coup d'envoi)
  * `event_sec`: seconde de l'événement (par rapport au coup d'envoi)
  * `event_seconds_elapsed`: nombre de seconde écoulé depuis le coup d'envoi
  * `event_player_id`: identifiant du joueur
  * `event_team_id`: identifiant de l'équipe
  * `event_x`: position en abscisse de l'événement 
  * `event_y`: position en ordonnée de l'événement
  * `zone_id`: identifiant de la zone
  * `zone_name`: nom de la zone détaillée
  * `big_zone_name`: nom de la zone large
  * `event_order`: ordre des événements entre eux


* Données de joueurs avec les indicateurs de temps de jeu: `players.csv`:
  * `player_id`: identifiant du joueur (dérivé de `uid`)
  * `teamName`: 
  * `name`: 
  * `position`: 
  * `birth_date`: 
  * `birth_place`: 
  * `country`: 
  * `deceased`: 
  * `first_name`: 
  * `first_nationality`: 
  * `height`: 
  * `jersey_num`: 
  * `join_date`: 
  * `known_name`: 
  * `last_name`: 
  * `leave_date`: 
  * `middle_name`: 
  * `new_team`: 
  * `preferred_foot`: 
  * `real_position`: 
  * `real_position_side`: 
  * `weight`: 
  * `loan`:
  * `uid`: 
  * `minutes_duration`: temps de jeu sur les données d'apprentissage
  * `joueur_etudie`: 1 si le joueur est susceptible d'être éudié, 0 sinon (problématique non traité sur le challenge Lincoln)
* Dictionnaire de zone en csv: `zones_dict.csv`:
  * `xmin` : coordonnées min en abscisse de la boite
  * `xmax` : coordonnées max en abscisse de la boite
  * `ymin` : coordonnées min en ordonnée de la boite
  * `ymax` : coordonnées max en ordonnée de la boite
  * `zone_id` : identifiant de la zone
  * `zone_name` : nom de la zone détaillée
  * `big_zone_name` : nom de la zone large