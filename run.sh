#!/bin/bash
# Script de lancement du serveur voice assistant

echo "🎯 Démarrage du serveur Voice Assistant Python"

# Utiliser le Python du venv local directement (évite les conflits d'env)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Vérifier si les dépendances sont installées
if ! $PYTHON -c "import aiohttp" 2>/dev/null; then
    echo "📦 Installation des dépendances..."
    $PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Lancer le serveur
$PYTHON voice_server.py