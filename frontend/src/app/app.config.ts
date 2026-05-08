import { ApplicationConfig } from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';

export const appConfig: ApplicationConfig = {
  // Le agregamos withFetch() aquí adentro
  providers: [provideHttpClient(withFetch())]
};
