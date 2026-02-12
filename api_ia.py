import re
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    Groq = None


class ApiIa:
    # Generar funcion para preguntar al modelo de IA sobre el contenido de un texto
    PROMPT_IA_RESUMEN = '''
    Te entregaré un texto extraido de un documento en base al rubro de la construccion
    En base a este documento necesito que me generes lo siguiente, con estructura en markdown:

    #### Resultado
    #### 🧾 Resumen
    [Aqui genera un resumen de que trata el documento, maximo 7 lineas]

    #### 📋 Requisitos / frases detectadas
    [Aqui genera una lista desordenada de los requisitos y frases clave del rubro detectadas en el documento , máximo 10 list elements]

    #### ✅ Checklist QA/QC
    [Aqui genera un texto simple, con los elementos checklist QA/QC que hayas detectado, usando el siguiente formato:
     [item del checklist 1 que generes ~~ item2 ~~ item3 ~~ etc], minimo 6 items y máximo 20. Fijate que todos los items que generes 
     deben ir dentro de corchetes [] y separados por doble virgulilla ~~ . OJO no deben ser por separado, un ejemplo de como 
     debes entregarlo seria: [tarea1~~tarea2~~tarea3~~etc]
    ]

    Necesito que lo generes tal cual con esa estructura, no mas, no menos. Se breve.
    Además, revisa siempre que todo el texto este en idioma español.
    A continuación te entrego el texto sobre el cual debes generar lo solicitado:
    
   '''

    def __init__(self, client_groq):
        self.client_groq = client_groq  

    def generate_ia_resume(self,text_document: str,
                            model="openai/gpt-oss-20b",
                            temperature=0.3,
                            max_completion_tokens=2000,
                            ):
        '''
        ########## PARAMETROS DE LA FUNCION EXPLICADOS ###########
        text_document: El texto extraido del documento de la biblioteca EETT que se le agregara al prompt declarado
        model : La API key de groq disponibiliza varios modelos de IA, de momento estamos usando el modelo de openai, pero se puede cambiar en un futuro
        temperature: Determina que tan creativa sera la respuesta del modelo, el rango de 0.1 a 0.3 es determinista, no se necesita ser creativo para el prompt que se le pasa,
        max_completion_tokens= Determina cuantos caracteres como máximo generará la respuesta, para ahorrar en cuota de uso de la API se usan 1600 que equivalen a unos 4800 caracteres aprox ,
        '''

        texto_seguro = text_document.strip()[:18000]

        # El rol indica un contexto para la IA, de donde viene cada instruccion, el prompt es sistematico, y el mensaje variable
        messages = [
            {"role": "system", "content": self.PROMPT_IA_RESUMEN.strip()},
            {"role": "user", "content": texto_seguro}
        ]
        try:
            
            response = self.client_groq.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_completion_tokens,
                stream=False,
                # Stream define la forma en que la IA responde, si lo ponemos en TRUE, entregara caracter por caracter tipo "chat", como no es lo que buscamos, sino una respuesta completa, queda en false
            )

            # Extrae el texto final y lo devuelve
            return response.choices[0].message.content or ""
        except Exception as e:
            error_msg = str(e)
            if"Limit" in error_msg or "413" in error_msg:
                return "El documento es demaciado grande para la version gratuita de IA. \n Se intento procesar pero se excedio el limite"
            return f"Error al consultar a la IA: {error_msg}"

    # Objetivo de la funcion: Guardar el resumen generado de la ia en base a un documento, para ahorrar tokens, y tener consistencia
    def save_resume_ia(self,resume_generated, id_documento):
        try:
            with open("qintegrity_resumen_ia.txt", "a", encoding="utf-8") as f:
                contenido = f"INIT_CONTENT-{id_documento.strip()}\n{resume_generated}\nEND_CONTENT-{id_documento.strip()}\n"
                f.write(contenido)
                return f"✅ Guardada respuesta IA"
        except FileNotFoundError:
            return ("❌ El archivo no existe")
        except PermissionError:
            return ("❌ No tienes permisos para leer el archivo")
        except UnicodeDecodeError:
            return ("❌ Problema de codificación del archivo")
        except Exception as e:
            return (f"❌ Error inesperado: {e}")

    def check_resumen_ia(self,id_documento):
        try:
            respuestas_ia = Path("qintegrity_resumen_ia.txt").read_text(encoding="utf-8")
            if id_documento in respuestas_ia:
                pat = rf"(?s)INIT_CONTENT-{re.escape(id_documento)}\r?\n(.*?)\r?\nEND_CONTENT-{re.escape(id_documento)}"
                m = re.search(pat, respuestas_ia)
                contenido = m.group(1) if m else None
                return contenido


            else:
                return ""

        except FileNotFoundError:
            return ("❌ El archivo no existe")
        except PermissionError:
            return ("❌ No tienes permisos para leer el archivo")
        except UnicodeDecodeError:
            return ("❌ Problema de codificación del archivo")
        except Exception as e:
            return (f"❌ Error inesperado: {e}")

    def generate_checkboxes(self, ia_resume: str) -> list[str]:
        if not ia_resume:
            return ["No se pudo generar el resumen IA"]

        chk_pattern = r"(?sm)\[(.*?)\]"  # contenido dentro de [ ... ] (multiline)
        m = re.search(chk_pattern, ia_resume)
        if not m:
            return ["No se encontraron checkboxes"]

        raw = (m.group(1) or "").strip()
        if not raw:
            return ["No se encontraron checkboxes"]

        items = [x.strip() for x in raw.split("~~")]
        return [x for x in items if x]  # sin vacíos

    def clean_checkboxes(self, ia_resume: str) -> str:
        if not ia_resume:
            return "No se pudo generar el resumen IA"

        chk_pattern = r"(?sm)\[(.*?)\]"
        # Elimina el bloque [ ... ] completo (y lo que contenga)
        return re.sub(chk_pattern, "", ia_resume).strip()
    
    def chat_interactivo(self, mensaje_usuario, historial_mensajes, ia_contenido):
        
        model = "openai/gpt-oss-120b"
        # Limitamos el contexto para no exceder la ventana de tokens (aprox 12k chars de seguridad)
        contexto = str(ia_contenido)
        
        # Definimos el comportamiento: la IA DEBE responder en base al contexto
        system_prompt = (
            "Eres un experto Auditor de Construcción y Control de Calidad. "
            "Tu tarea es responder dudas basándote EXCLUSIVAMENTE en la siguiente Especificación Técnica (EETT):\n\n"
            f"--- INICIO DOCUMENTO ---\n{contexto}\n--- FIN DOCUMENTO ---\n\n"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Añadimos el historial previo (memoria del chat)
        for m in historial_mensajes:
            messages.append(m)
            
        # Añadimos la pregunta actual
        messages.append({"role": "user", "content": mensaje_usuario})

        response = self.client_groq.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2 # Baja temperatura para evitar alucinaciones
        )
        return response.choices[0].message.content
        
