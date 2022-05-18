import json
from os import path


header_comment = '# %%\n'

class JupyteraPython:

    def __init__(self):
        self

    def nb2py(notebook):
        """
        Método encargado de transformar un archivo .ypnb a un archivo .py
        Returns:

        """
        result = []
        cells = notebook['cells']

        for cell in cells:
            cell_type = cell['cell_type']

            if cell_type == 'markdown':
                result.append('%s"""\n%s\n"""' %
                              (header_comment, ''.join(cell['source'])))

            if cell_type == 'code':
                cadena=[]
                for i in cell['source']:
                    if i.startswith('!') or i.startswith('%'):
                        continue
                    else:
                        resultado=i.find("print")
                        cadena_sinespacios=i.strip()
                        if(resultado!=-1 and not cadena_sinespacios.startswith("#") and not cadena_sinespacios.startswith("def")
                            and not cadena_sinespacios.startswith("from") and cadena_sinespacios!='print'):
                            if(i[resultado+len("print")]!="(" and i[resultado+len("print")]!="_"):
                                    cadenanueva=i[:resultado+len("print")]+"("
                                    comentario=i[resultado + len("print"):].find("#")
                                    funcionformat=i[resultado + len("print"):].find(".format")
                                    if(comentario!=-1):
                                        cuerpo = i[resultado + len("print"):comentario]
                                        cuerpo_coment=i[comentario:]
                                        cadena_final=cadenanueva+cuerpo+cuerpo_coment+")"+"\n"
                                        cadena.append(cadena_final)
                                    else:
                                        if(funcionformat!=-1):
                                            if(comentario==-1):
                                                cuerpo = i[resultado + len("print"):]
                                            else:
                                                cuerpo = i[resultado + len("print"):comentario]
                                            cadena_final = cadenanueva + cuerpo + ")"+"\n"
                                            cadena.append(cadena_final)
                                        else:
                                            cuerpo=i[resultado+len("print"):]+")"
                                            cadena_final=cadenanueva+cuerpo+"\n"
                                            cadena.append(cadena_final)
                            else:
                                if(i[resultado+len("print")]=="("):
                                    cadena.append(i)
                        else:
                            if(cadena_sinespacios!="print"):
                                cadena.append(i)

                result.append("%s%s" % (header_comment, ''.join(cadena)))
        return '\n\n'.join(result)

    def py2nb(py_str):
        # remove leading header comment
        if py_str.startswith(header_comment):
            py_str = py_str[len(header_comment):]

        cells = []
        chunks = py_str.split('\n\n%s' % header_comment)

        for chunk in chunks:
            cell_type = 'code'
            if chunk.startswith("'''"):
                chunk = chunk.strip("'\n")
                cell_type = 'markdown'
            elif chunk.startswith('"""'):
                chunk = chunk.strip('"\n')
                cell_type = 'markdown'

            cell = {
                'cell_type': cell_type,
                'metadata': {},
                'source': chunk.splitlines(True),
            }

            if cell_type == 'code':
                cell.update({'outputs': [], 'execution_count': None})

            cells.append(cell)

        notebook = {
            'cells': cells,
            'metadata': {
                'anaconda-cloud': {},
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'},
                'language_info': {
                    'codemirror_mode': {'name': 'ipython', 'version': 3},
                    'file_extension': '.py',
                    'mimetype': 'text/x-python',
                    'name': 'python',
                    'nbconvert_exporter': 'python',
                    'pygments_lexer': 'ipython3',
                    'version': '3.6.1'}},
            'nbformat': 4,
            'nbformat_minor': 4
        }

        return notebook

    def convert(in_file, out_file):

        _, in_ext = path.splitext(in_file)
        _, out_ext = path.splitext(out_file)

        if in_ext == '.ipynb' and out_ext == '.py':
            with open(in_file, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            py_str = JupyteraPython.nb2py(notebook)
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(py_str)

        elif in_ext == '.py' and out_ext == '.ipynb':
            with open(in_file, 'r', encoding='utf-8') as f:
                py_str = f.read()
            notebook = JupyteraPython.py2nb(py_str)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2)


        else:
            raise (Exception('Extensions must be .ipynb and .py or vice versa'))