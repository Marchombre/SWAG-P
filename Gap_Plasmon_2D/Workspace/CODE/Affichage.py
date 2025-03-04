# Tracé du graphe de réflectance Rup en fonction de la longueur d'onde
plt.figure(figsize=(8, 5))
plt.plot(lambda_range, Ru_material_test, label=f"Au thick : {int(thick_metal)} nm")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (Rup)")
plt.title("Réflectance R_up en fonction de la longueur d'onde")
plt.legend()
plt.grid(True)
plt.savefig("material_test_Rup.jpg")
plt.show(block=False)

# Sauvegarde des résultats
R = [Ru_material_test, Rd_material_test]
np.savez("data_accroches_all_Rdown-Rup.npz", list_wavelength=lambda_range, R=R)
