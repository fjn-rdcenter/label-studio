import React from "react";
import { SidebarMenu } from "../../components/SidebarMenu/SidebarMenu";
import { WebhookPage } from "../WebhookPage/WebhookPage";
import { DangerZone } from "./DangerZone";
import { GeneralSettings } from "./GeneralSettings";
import { AnnotationSettings } from "./AnnotationSettings";
import { LabelingSettings } from "./LabelingSettings";
import { MachineLearningSettings } from "./MachineLearningSettings/MachineLearningSettings";
import { MembersSettings } from "./MembersSettings/MembersSettings";
import { PredictionsSettings } from "./PredictionsSettings/PredictionsSettings";
import { StorageSettings } from "./StorageSettings/StorageSettings";
import { isInLicense, LF_CLOUD_STORAGE_FOR_MANAGERS } from "../../utils/license-flags";

const isAllowCloudStorage = !isInLicense(LF_CLOUD_STORAGE_FOR_MANAGERS);

export const MenuLayout = ({ children, ...routeProps }) => {
  return (
    <SidebarMenu
      menuItems={[
        GeneralSettings,
        MembersSettings,
        LabelingSettings,
        AnnotationSettings,
        MachineLearningSettings,
        PredictionsSettings,
        isAllowCloudStorage && StorageSettings,
        WebhookPage,
        DangerZone,
      ].filter(Boolean)}
      path={routeProps.match.url}
      children={children}
    />
  );
};

const pages = {
  MembersSettings,
  AnnotationSettings,
  LabelingSettings,
  MachineLearningSettings,
  PredictionsSettings,
  WebhookPage,
  DangerZone,
};

isAllowCloudStorage && (pages.StorageSettings = StorageSettings);

export const SettingsPage = {
  title: "Settings",
  path: "/settings",
  exact: true,
  layout: MenuLayout,
  component: GeneralSettings,
  pages,
};
