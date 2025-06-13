from graphviz import Digraph

def generar_diagrama_clases(output_file="diagrama_clases"):
    """
    Genera un diagrama de clases básico de la jerarquía Vehiculo -> Auto, Moto.
    El archivo se guarda como PDF y PNG.
    """
    dot = Digraph(comment="Diagrama de Clases - Vehiculos")

    # Nodo base
    dot.node('Vehiculo', 'Vehiculo\n+ marca\n+ modelo\n+ año\n+ obtener_info()')

    # Clases hijas
    dot.node('Auto', 'Auto\n+ abrir_maletero()')
    dot.node('Moto', 'Moto\n+ hacer_caballito()')

    # Relaciones
    dot.edge('Vehiculo', 'Auto')
    dot.edge('Vehiculo', 'Moto')

    # Guardar
    dot.render(filename=output_file, format='png', cleanup=True)
    print(f"📄 Diagrama generado en: {output_file}.png")

