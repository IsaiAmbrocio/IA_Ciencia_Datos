from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Cargar el modelo de clasificación previamente entrenado
modelo = joblib.load('modelo_clasificacion.pkl')

@app.route('/')
def index():
    return render_template('app.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de entrada del usuario
        data = request.form['features']

        # Realizar una predicción con el modelo cargado
        prediction = modelo.predict([data])[0]

        # Renderizar la página HTML con la predicción
        return render_template('app.html', prediction=prediction)

    except Exception as e:
        return render_template('app.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

