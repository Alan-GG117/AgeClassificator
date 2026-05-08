import { Component, inject, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class AppComponent {
  http = inject(HttpClient);
  cdr = inject(ChangeDetectorRef);

  archivoSeleccionado: File | null = null;
  vistaPreviaImagen: string | ArrayBuffer | null = null;
  resultadoJson: any = null;
  cargando: boolean = false;

  // Se ejecuta cuando el usuario elige una foto
  alSeleccionarArchivo(event: any) {
    const archivo = event.target.files[0];
    if (archivo) {
      this.archivoSeleccionado = archivo;
      this.resultadoJson = null; // Limpiar resultados anteriores

      // Crear vista previa para mostrar la imagen en pantalla
      const reader = new FileReader();
      reader.onload = e => this.vistaPreviaImagen = reader?.result as string;
      reader.readAsDataURL(archivo);
    }
  }

  // Se ejecuta al dar clic en el botón
  clasificarEdad() {
    if (!this.archivoSeleccionado) return;

    this.cargando = true;
    const formData = new FormData();
    // 'image' debe coincidir exactamente con el parámetro de tu endpoint en FastAPI
    formData.append('image', this.archivoSeleccionado);

    // Hacemos la petición POST a tu servidor local
    // Hacemos la petición POST a tu servidor local
    this.http.post('http://127.0.0.1:8000/age_classifier', formData).subscribe({
      next: (respuesta) => {
        console.log("¡ÉXITO! LLEGÓ LA RESPUESTA: ", respuesta); // <- Rastreador 1
        this.resultadoJson = respuesta;
        this.cargando = false;
        this.cdr.detectChanges();
      },
      error: (error) => {
        console.error("¡ERROR CAPTURADO POR ANGULAR!: ", error); // <- Rastreador 2
        alert("Hubo un error de conexión. Revisa la consola (F12).");
        this.cargando = false;
        this.cdr.detectChanges();
      }
    });
  }
}
