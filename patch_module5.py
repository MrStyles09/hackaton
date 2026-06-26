"""
PATCH MODULE 5 — Recalibrage prototypes + seuils
=================================================
A lancer UNE FOIS dans ton dossier de travail :
    python patch_module5.py

Ce script modifie directement module5_criticite.py pour :
1. Abaisser les seuils ROUGE/ORANGE à la distribution réelle (0.0-0.06)
2. Enrichir les prototypes avec un style proche des proverbes mooré
3. Ajouter une dimension "sagesse_menace" adaptée aux proverbes
4. Rescorer les 1159 segments avec les nouveaux paramètres
"""

import re, os, sys

TARGET = "module5_criticite.py"
if not os.path.exists(TARGET):
    print(f"ERREUR : {TARGET} introuvable dans le dossier courant.")
    print("Lance ce script depuis ton dossier CITA'SC 2026.")
    sys.exit(1)

with open(TARGET, encoding="utf-8") as f:
    content = f.read()

# ── 1. Remplacer les prototypes ──────────────────────────────────────────────

OLD_PROTOTYPES = 'CRITICITE_PROTOTYPES = {'
NEW_PROTOTYPES = '''CRITICITE_PROTOTYPES = {
    # NOTES SUR LA RECALIBRATION :
    # Le corpus est composé de proverbes mooré (style indirect, métaphorique).
    # Les prototypes originaux ciblaient des messages d'urgence explicites.
    # Recalibration : prototypes enrichis avec vocabulaire indirect + métaphores
    # + phrases proches du style proverbe pour couvrir les signaux implicites.

    "urgence_sanitaire": [
        # Urgence explicite
        "des gens sont malades dans notre village",
        "il y a une épidémie de paludisme",
        "les médicaments sont en rupture de stock au centre de santé",
        "un enfant est mort de la rougeole",
        "nous avons besoin d'aide médicale urgente",
        "cas de choléra signalé dans le quartier",
        "la femme a accouché sans sage-femme il y a un problème",
        # Style indirect / proverbe
        "celui qui est malade ne peut pas attendre demain pour chercher un remède",
        "quand la maladie entre dans une maison tout le village en souffre",
        "le corps qui souffre n'a pas de patience",
        "la mort frappe sans prévenir les familles sans médicament",
        "soigner un enfant malade est plus urgent que tout le reste",
        # Mooré
        "ned sãame ne zĩnga pãnga",
        "rogom sẽed n tɩ pɛɛg ka be",
        "bãngr soaba sãame",
    ],
    "tension_sociale": [
        # Tension explicite
        "il y a des tensions entre les communautés",
        "des gens appellent à manifester contre les autorités",
        "une rumeur circule sur cet homme qui aurait fait quelque chose",
        "des jeunes ont bloqué la route avec des barrières",
        "conflit entre agriculteurs et éleveurs dans la zone",
        "on accuse le chef du village d'avoir volé l'argent",
        "des tirs ont été entendus cette nuit au village",
        # Style indirect / proverbe
        "deux chefs dans un même village c'est la guerre assurée",
        "quand les voisins se disputent c'est toute la communauté qui perd",
        "l'injustice d'un seul devient la colère de tous",
        "celui qui provoque le feu ne contrôle pas où il brûle",
        "la discorde entre frères est plus dangereuse que l'ennemi extérieur",
        "le village divisé ne résiste pas à l'adversité",
        # Mooré
        "yãmb yaa fo sẽed zĩnga",
    ],
    "alerte_agricole": [
        # Alerte explicite
        "les criquets ont envahi les champs de mil",
        "la sécheresse menace la récolte cette année",
        "les semences d'engrais ne sont pas disponibles",
        "attaque acridienne dans la région",
        "les pluies tardent et les champs sont secs",
        "récolte catastrophique à cause du manque de pluie",
        "les animaux meurent de soif et de faim",
        # Style indirect / proverbe
        "quand la pluie ne vient pas les greniers restent vides",
        "le paysan qui perd sa récolte perd toute une année de sa vie",
        "la terre asséchée ne nourrit pas ses enfants",
        "si les semailles échouent la faim arrive avant la saison sèche",
        "l'animal qui ne trouve pas à boire ne peut pas labourer",
        "la récolte perdue c'est la famille qui souffre pendant des mois",
    ],
    "desinformation": [
        # Désinformation explicite
        "c'est un mensonge ce qu'on dit sur le vaccin",
        "cette information est fausse elle a été inventée",
        "on raconte des choses fausses pour provoquer la peur",
        "cette rumeur n'est pas vraie j'ai vérifié",
        "les gens diffusent de fausses nouvelles sur la situation sécuritaire",
        "quelqu'un a inventé cette histoire pour nuire",
        # Style indirect / proverbe
        "la vérité finit toujours par rattraper le mensonge même après longtemps",
        "celui qui ment une fois ne sera plus cru même quand il dit la vérité",
        "une rumeur propagée vite fait plus de dégâts que la vérité tardive",
        "les fausses paroles voyagent plus vite que la réalité",
        "la langue qui ment blesse plus profondément que le couteau",
        "distinguer le vrai du faux est la sagesse la plus difficile",
    ],
    "detresse_individu": [
        # Détresse explicite
        "je suis seul et j'ai besoin d'aide urgent",
        "ma famille est en danger nous n'avons plus rien à manger",
        "nous sommes bloqués et nous n'arrivons pas à partir",
        "s'il vous plaît aidez-nous nous avons besoin de secours",
        "des hommes armés sont venus et ont tout pris",
        "nous avons fui et nous n'avons nulle part où aller",
        # Style indirect / proverbe
        "l'homme seul dans l'adversité est comme un arbre sans racines face au vent",
        "celui qui n'a personne pour l'aider dans le malheur est vraiment perdu",
        "sans famille sans soutien l'être humain ne peut pas surmonter l'épreuve",
        "appeler à l'aide et ne pas être entendu c'est la pire des souffrances",
        "quand quelqu'un est en danger tout le village a le devoir d'agir",
    ],
    "sagesse_menace": [
        # Dimension NOUVELLE : proverbes portant un avertissement implicite
        # Couvre les messages d'alerte formulés de façon indirecte
        "il vaut mieux prévenir que guérir",
        "celui qui ignore les signes avant-coureurs sera surpris par le danger",
        "le sage voit le danger venir avant qu'il n'arrive",
        "mieux vaut se préparer avant que le malheur ne frappe",
        "n'attends pas que le feu brûle ta maison pour chercher de l'eau",
        "l'avertissement ignoré devient une catastrophe annoncée",
        "celui qui ne voit pas venir le danger est en plus grand danger encore",
        # Mooré style
        "Baag ka wãbd gãngãogẽ n le yaas n ges reem ye",
        "sẽn togs-a zõang ma kʋʋr n gãt-a nugu",
    ],
}'''

if OLD_PROTOTYPES in content:
    # Trouver la fin du bloc CRITICITE_PROTOTYPES actuel
    start = content.index(OLD_PROTOTYPES)
    # Trouver la fin : ligne "}" seule après le bloc
    depth = 0
    end = start
    for i, ch in enumerate(content[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    content = content[:start] + NEW_PROTOTYPES + content[end:]
    print("OK Prototypes enrichis (6 dimensions dont sagesse_menace)")
else:
    print("ATTENTION : bloc CRITICITE_PROTOTYPES non trouvé — vérifier le fichier")

# ── 2. Remplacer les SEUILS ──────────────────────────────────────────────────

OLD_SEUILS = '''SEUILS = {
    "urgence_sanitaire" : 0.45,
    "tension_sociale"   : 0.42,
    "alerte_agricole"   : 0.40,
    "desinformation"    : 0.38,
    "detresse_individu" : 0.44,
}'''

NEW_SEUILS = '''# Seuils recalibrés sur la distribution réelle du corpus mooré (0.0-0.06)
# Anciens seuils : 0.38-0.45 (calibrés pour messages d'urgence explicites)
# Nouveaux seuils : 0.03-0.05 (calibrés sur scores observés 0.0-0.06)
SEUILS = {
    "urgence_sanitaire" : 0.035,
    "tension_sociale"   : 0.030,
    "alerte_agricole"   : 0.028,
    "desinformation"    : 0.025,
    "detresse_individu" : 0.035,
    "sagesse_menace"    : 0.040,
}'''

if OLD_SEUILS in content:
    content = content.replace(OLD_SEUILS, NEW_SEUILS)
    print("OK Seuils recalibrés (0.025-0.040)")
else:
    # Essai avec regex au cas où l'espacement diffère
    pattern = r'SEUILS\s*=\s*\{[^}]+\}'
    match = re.search(pattern, content)
    if match:
        content = content[:match.start()] + NEW_SEUILS + content[match.end():]
        print("OK Seuils recalibrés via regex")
    else:
        print("ATTENTION : bloc SEUILS non trouvé — patch manuel nécessaire")

# ── 3. Recalibrer les niveaux ROUGE/ORANGE ───────────────────────────────────

OLD_NIVEAUX = '''        if score_max >= 0.65 or len(alertes) >= 2:
            niveau = "ROUGE"
        elif score_max >= 0.45 or len(alertes) == 1:
            niveau = "ORANGE"'''

NEW_NIVEAUX = '''        # Seuils de niveau recalibrés sur distribution corpus mooré (0.0-0.06)
        if score_max >= 0.055 or len(alertes) >= 2:
            niveau = "ROUGE"
        elif score_max >= 0.035 or len(alertes) == 1:
            niveau = "ORANGE"'''

if OLD_NIVEAUX in content:
    content = content.replace(OLD_NIVEAUX, NEW_NIVEAUX)
    print("OK Niveaux ROUGE/ORANGE recalibrés (0.055/0.035)")
else:
    print("ATTENTION : bloc niveaux non trouvé — les seuils ROUGE/ORANGE restent inchangés")

# ── 4. Ajouter sagesse_menace dans COULEURS ──────────────────────────────────

OLD_COULEURS = '''COULEURS = {
    "urgence_sanitaire" : "#E53E3E",   # rouge
    "tension_sociale"   : "#ED8936",   # orange
    "alerte_agricole"   : "#38A169",   # vert
    "desinformation"    : "#805AD5",   # violet
    "detresse_individu" : "#3182CE",   # bleu
}'''

NEW_COULEURS = '''COULEURS = {
    "urgence_sanitaire" : "#E53E3E",   # rouge
    "tension_sociale"   : "#ED8936",   # orange
    "alerte_agricole"   : "#38A169",   # vert
    "desinformation"    : "#805AD5",   # violet
    "detresse_individu" : "#3182CE",   # bleu
    "sagesse_menace"    : "#D69E2E",   # or — dimension proverbe
}'''

if OLD_COULEURS in content:
    content = content.replace(OLD_COULEURS, NEW_COULEURS)
    print("OK Couleur sagesse_menace ajoutée")
else:
    print("INFO : bloc COULEURS non trouvé — couleur sagesse_menace non ajoutée (non bloquant)")

# ── 5. Écrire le fichier patché ───────────────────────────────────────────────

backup = TARGET + ".backup"
import shutil
shutil.copy(TARGET, backup)
print(f"Backup créé : {backup}")

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nOK {TARGET} patché avec succès.")
print("""
Lance maintenant :

  python module5_criticite.py ^
    --json corpus/segments.json ^
    --mode zeroshot ^
    --db corpus/metadata.db

Puis vérifie les scores :

  python -c "
import json, sqlite3
conn = sqlite3.connect('corpus/metadata.db')
rows = conn.execute('''
    SELECT transcription, translation_fr, criticite_json
    FROM segments
    WHERE criticite_json IS NOT NULL
    ORDER BY json_extract(criticite_json, '$.score_max') DESC
    LIMIT 10
''').fetchall()
for r in rows:
    import json as j
    crit = j.loads(r[2])
    print(crit['niveau'], crit['score_max'], '|', (r[1] or '')[:60])
conn.close()
"
""")
