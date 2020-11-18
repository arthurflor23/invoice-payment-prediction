define([
    'qlik',
    './properties',
    'text!./hello-world.ng.html',
    'css!./hello-world.ng.css'
  ],
  function (qlik, props, template, style) {
    'use strict';

    return {
      definition: props,
      template: template,
      style: style,
      controller: ['$scope', '$compile', function ($scope, $compile) {
        $scope.initElements = async function (key) {
          await (async (k) => $scope.key = k)(key);
          $('#qs-toolbar-container').append($(`#${key}-navbar`));
        }

        $scope.destroyElements = function () {
          $scope.component.model.Validated.unbind();
          angular.forEach($(`[id*='${$scope.key}']`), (elem) => $(elem).remove());
          angular.forEach($(`[class*='${$scope.key}']`), (elem) => {
            $(elem).removeClass((_, style) => {
              return style.split(' ').filter((w) => w.includes($scope.key)).join(' ');
            });
          });
        }

        $scope.changePath = function (path, option) {
          var dividor = '/state/';
          var index = window.location.pathname.indexOf(dividor);
          var pathname = window.location.pathname.substring(0, index + dividor.length);
          window.open(pathname + path, option);
        }

        // $scope.component.model.Validated.bind(() => $scope.initElements());
        $scope.$on('$destroy', () => $scope.destroyElements());
        $scope.initElements(props.key);


        // qlik.navigation.prevSheet();
        // qlik.navigation.nextSheet();
        // qlik.navigation.setMode('edit');
        // qlik.navigation.setMode('analysis');



        // var element = '<div><div role="dialog" class="lui-popover" id="rlui-popover-1" style="visibility: visible; position: absolute; max-width: 500px; top: 44.5px; left: 10px;"><div><ul class="lui-list qs-toolbar__menu" data-tid="nav-menu" role="menu"><div><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="4dfdd3" title="App overview" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--home" aria-hidden="true"></span></div><div class="lui-list__text">App overview</div><div class="list-divider"></div><div class="lui-list__aside  lui-nopad"><button type="button" class="lui-fade-button" title="Open in new tab" tabindex="-1"><span class="lui-icon lui-icon--new-tab lui-fade-button__icon" aria-hidden="true" style="text-align: center; margin: auto;"></span></button></div></li><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="4dfdd2" title="Open hub in a new tab" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--hub-logo" aria-hidden="true"></span></div><div class="lui-list__text">Open hub</div><div class="lui-list__aside  lui-nopad"><button type="button" class="lui-fade-button" title="Open in new tab" tabindex="-1"><span class="lui-icon lui-icon--new-tab lui-fade-button__icon" aria-hidden="true" style="text-align: center; margin: auto;"></span></button></div></li></div><div class="qs-toolbar__divider"></div><div><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="GlobalMenu.ExportSheetToPdf" title="Download sheet as PDF" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--download" aria-hidden="true"></span></div><div class="lui-list__text">Download sheet as PDF</div></li><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="GlobalMenu.DuplicateSheet" title="Duplicate sheet" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--copy" aria-hidden="true"></span></div><div class="lui-list__text">Duplicate sheet</div></li><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="GlobalMenu.DeleteSheet" title="Delete sheet" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--bin" aria-hidden="true"></span></div><div class="lui-list__text">Delete sheet</div></li><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="i9fd7" title="Embed sheet" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--code" aria-hidden="true"></span></div><div class="lui-list__text">Embed sheet</div></li><li class="lui-list__item qs-toolbar__menu-item no-hover" role="menuitem" tid="4dfe29" title="Touch screen mode" tabindex="-1"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--touch" aria-hidden="true"></span></div><div class="lui-list__text"><div class="qs-toolbar__toggle-touch"><div class="qs-toolbar__toggle-touch-label">Touch screen mode<strong>Off</strong></div><div class="qs-toolbar__toggle-touch-switch"><div class="lui-switch"><label class="lui-switch__label"><input class="lui-switch__checkbox" type="checkbox"><span class="lui-switch__wrap"><div class="lui-switch__inner"></div><div class="lui-switch__switch"></div></span></label></div></div></div></div></li><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="4dfe26" title="Open the help site in new tab" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--help" aria-hidden="true"></span></div><div class="lui-list__text">Help</div></li><li class="lui-list__item qs-toolbar__menu-item" role="menuitem" tid="4dfe27" title="About" tabindex="0"><div class="lui-list__aside  qs-toolbar__menu-item-icon"><span class="lui-icon lui-icon--info" aria-hidden="true"></span></div><div class="lui-list__text">About</div></li></div></ul></div><div class="lui-popover__arrow  lui-popover__arrow--top" style="left: 30px;"></div></div></div>'
        // $compile(element)($scope);
        // angular.element('.qs-popover-container.qs-toolbar__element').append(element);

        // var pElement = angular.element('#qs-toolbar-container');
        // var pElementCopy = pElement.clone();
        // $compile(pElementCopy)($scope);
        // angular.element('.qs-header').append(pElementCopy);


        // $('.qs-header').append($compile($(`#qs-toolbar-container`).clone())($scope));
        // $('#qs-toolbar-container').append($compile($(`[tid='2f7a7e']`).clone())($scope));


        // console.log(qlik.navigation);
        // console.log(qlik.navigation.getCurrentSheetId());
        console.log($scope.layout);
        // console.log(qlik);
      }]
    };
  });