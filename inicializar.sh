#bin/bash
echo "Inicializando entorno virtual de Python"
python3.6 -m venv venv
echo "Activando entorno virtual"
source venv/bin/activate
echo "Actualizando gestor de paquetes de Python (pip)"
pip install --upgrade pip
echo "Instalando paquetes necesarios (ver requirements.txt)"
pip install -r requirements.txt
echo "Listo. Active manualmente entorno virtual con el comando 'source venv/bin/activate"
