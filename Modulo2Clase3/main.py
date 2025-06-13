from diagram import generar_diagrama_clases
from vehiculos import Auto, Moto, VehiculoJSONPrinter, VehiculoPrinter
from utils import mostrar_menu

# Lista de vehículos registrados
vehiculos = [
    Auto("Toyota", "Corolla", 2020),
    Moto("Yamaha", "R3", 2021)
]

while True:
    mostrar_menu()
    opcion = input("Seleccione una opción: ")

    if opcion == "1":
        print("\n📦 Vehículos registrados:")
        for i, v in enumerate(vehiculos):
                print(f"{i+1}.", end=" ")
                VehiculoPrinter.mostrar_info(v)

    elif opcion == "2":
        print("\n🏁 Acelerando vehículos:")
        for v in vehiculos:
            v.acelerar()

    elif opcion == "3":
        print("\n🔧 Acciones especiales:")
        for v in vehiculos:
            if isinstance(v, Auto):
                v.abrir_maletero()
            elif isinstance(v, Moto):
                v.hacer_caballito()

    elif opcion == "4":
        print("\n💾 Exportando vehículos a JSON:")
        for i, v in enumerate(vehiculos):
            filename = f"vehiculo_{i+1}.json"
            VehiculoJSONPrinter.guardar_en_archivo(v, filename)
            print(f"✅ Vehículo exportado a {filename}")

    elif opcion == "5":
        generar_diagrama_clases()
    elif opcion == "0":
        print("👋 Saliendo del sistema.")
        break

    else:
        print("❌ Opción inválida. Intente nuevamente.")

