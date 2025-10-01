#!/usr/bin/env python3
"""Récupère les logs de l'ESP en temps réel"""

import asyncio
from aioesphomeapi import APIClient, LogLevel

async def main():
    print("🔌 Connexion à l'ESP...")

    api = APIClient("192.168.1.195", 6053, "", client_info="log-viewer")
    await api.connect(login=True)

    device_info = await api.device_info()
    print(f"✅ Connecté: {device_info.name}")
    print("📋 Logs en temps réel:\n")

    def on_log(log_entry):
        """Callback pour chaque ligne de log"""
        # Imprimer tous les attributs disponibles pour debugger
        print(f"LOG: {log_entry}", flush=True)

    # S'abonner aux logs (retourne directement un callback, pas une coroutine)
    print("📡 Abonnement aux logs...")
    unsubscribe = api.subscribe_logs(on_log, log_level=LogLevel.LOG_LEVEL_VERBOSE)

    print("✅ Abonné aux logs. Appuie sur Ctrl+C pour quitter.\n")

    # Garder la connexion ouverte
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Déconnexion...")
        await api.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
